# FlashAttention V1 实现与优化方案

> 状态：M1 完成（v0 朴素融合 kernel 实现，测试转绿；v1 online softmax 待实现） | 目标硬件：NVIDIA RTX 3060 Ti (sm_86) | 参考论文：*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (Dao et al., 2022)

## 1. 目标

在 `src/cuda/flashattention_v1/` 下从零实现 FlashAttention V1，遵循本仓库既有惯例（版本化 kernel 文件、GTest 正确性测试、独立 benchmark 可执行、CMake 子模块）。

- **正确性**：与 libtorch eager 参考实现对齐（fp32 下 `atol/rtol` 约 `1e-4` 量级）。
- **性能**：对常见形状（`N = 1024~4096`, `d = 64/128`），最终版本达到理论内存带宽的 80%~90% 量级。
- **可扩展**：接口预留精度模板（fp32 → fp16/TF32），causal 开关，后续可对比 FlashAttention-2。
- **过程可复现**：每个版本保留源码，性能演进记录到 `docs/`。

## 2. 硬件与环境约束

| 项目 | 数值 | 备注 |
|---|---|---|
| GPU | RTX 3060 Ti (GA104) | 8 GB GDDR6 |
| Compute Capability | 8.6 | Ampere，支持 `cp.async`、`mma.sync`、TF32 |
| CUDA | 13.0，`nvcc` | 目标架构 `sm_86`（顶层 CMake 已配置） |
| 显存带宽 | ~448 GB/s | roofline 关键参数 |
| FP32 | ~16 TFLOPS | CUDA core |
| FP16 Tensor Core | ~32 TFLOPS | 非稀疏，约为 FP32 的 2 倍 |
| TF32 Tensor Core | ~16 TFLOPS | 与 FP32 相当 |
| 寄存器 / SM | 65536 | 64K，影响每线程寄存器预算 |
| 共享内存 / SM | ~100 KB 量级 | 以 `cudaDeviceGetAttribute` 实测为准 |
| L2 Cache | ~4 MB 量级 | 可用于跨 CTA 的 K/V 复用 |

结论：**本 kernel 在 3060 Ti 上以内存带宽为主要瓶颈**，优化优先级是"减少并隐藏 HBM 访问"，Tensor Core 放在后面。

## 3. 算法回顾

标准注意力（单头，`Q, K, V ∈ R^{N×d}`）：

```
S = Q·Kᵀ / √d
P = softmax(S)            # 可选 causal mask
O = P·V
```

朴素实现需物化 `N×N` 的 S，HBM 流量 `O(N² + Nd)`。FlashAttention 将 KV 维度分块，配合 online softmax，把流量降为 `O(N²d²/M)`（M 为片上存储容量）。

Online softmax 核心状态（每行一组，全程 fp32）：

- `m`：running max
- `l`：running sum of exp
- `O`：加权累加

每处理一个 KV 块 `j` 的更新顺序：

1. 计算 `S_ij = Q_i·K_jᵀ / √d`（含 mask）；
2. `m_new = max(m, rowmax(S_ij))`；
3. `P̃ = exp(S_ij − m_new)`；
4. `l = l·exp(m − m_new) + rowsum(P̃)`；
5. `O = O·exp(m − m_new) + P̃·V_j`；
6. `m = m_new`。

循环结束后：`O = O / l`。

> 注意第 4、5 步必须先以新 max 完成 rescale，再累加新块，顺序颠倒会出错。

## 4. 接口与数据布局

采用**双层 API**：

- **params 入口（device 指针、无拷贝）**：`flash_attn_v0(const FlashAttentionParams&)`，直接按 params 中的 device 指针与 stride 启动 kernel，不做任何内存管理；
- **host 便捷包装（内部拷贝 + 建 params）**：`flash_attn_v0(const float* Q, const float* K, const float* V, float* O, int B, int H, int N, int d, bool causal)`，接收连续 row-major host 张量，内部拷贝进/出并构建 params，供 test/benchmark 使用。

kernel 模板为 `<T, AccT, HEAD_DIM, BLOCK_M, BLOCK_N, BLOCK_K, CAUSAL>`；运行时 `head_dim` 分派到 `HEAD_DIM ∈ {32, 64, 128}` 实例（96 走 128 实例）。版本演进中保持同一签名，后续 fp16 版本可用同名模板或新增 `flash_attn_v1_half`。

## 5. 项目集成

目标文件结构：

```text
src/cuda/flashattention_v1/
├── CMakeLists.txt
├── IMPLEMENTATION.md          # 本文件
├── flash_attn.cuh             # FlashAttentionParams + make_flash_attn_params
├── flashattention_v1.h        # 公共接口
├── flash_attn_v0.cu           # 朴素融合（正确性基准）
├── flash_attn_v1.cu           # online softmax 分块版
├── flash_attn_v2.cu           # tile 参数调优
├── flash_attn_v3.cu           # 向量化
├── flash_attn_v4.cu           # cp.async 流水
├── flash_attn_v5.cu           # Tensor Core
├── flash_attn_v6.cu           # causal 优化
├── flash_attn_v7.cu           # 细节打磨
├── test.cpp                   # GTest（可链接 libtorch 做参考）
└── benchmark.cu               # 性能测量
```

CMake 修改点：

1. 新建 `src/cuda/flashattention_v1/CMakeLists.txt`：静态库 + `test` + `benchmark` 可执行（参考 `src/cuda/gemm/CMakeLists.txt`，`USE_TORCH` 打开时链接 `/opt/libtorch` 用于参考实现）。
2. 在 `src/cuda/CMakeLists.txt` 中 `add_subdirectory(flashattention_v1)`。
3. 复用 `src/cuda/common/` 的 `benchmarkKernel`、`CHECK_CUDA`、随机初始化工具。

## 6. 实施里程碑

### M0：框架与参考实现

- 建好 CMake、头文件、测试与 benchmark 骨架（kernel 可先为空）。
- 用 libtorch 实现参考：`softmax((Q@Kᵀ)/√d + mask) @ V`，mask 填 `-inf`。
- 跑通"测试失败能定位、benchmark 能计时"的最小闭环。

**验收**：`cmake -DUSE_CUDA=ON -DUSE_TORCH=ON` 构建成功，测试框架可执行。

### M1：v0 朴素融合 kernel（正确性锚点）

设计：

- 一个 CTA 负责一个 Q tile（如 64 行）；
- 外层循环 KV 块，把 S 块算入 shared memory；
- 对每个 S 块做**传统两步 softmax**（先 max/sum，再归一化），累加进 O；
- 暂不引入 rescale，保证逻辑简单可调试。

**验收**：与 torch 参考对齐（fp32，`1e-4` 量级）；记录 baseline 性能；覆盖 `N ∈ {1, 64, 512, 1024, 2048}`、`d ∈ {64, 128}`、causal 开关。

### M2：v1 FlashAttention 核心（online softmax 分块）

设计：

- Q tile 常驻（寄存器或 shared），外层只循环 K/V 块（`B_c` 列）；
- 实现第 3 节的 6 步更新流程，`m/l/O` 全程 fp32；
- 最后 `O *= 1/l`；
- causal 时上三角填 `-inf`，且**先 mask 再求 max**。

**验收**：正确性与 v0 一致，性能显著优于 v0；`d` 非 8 倍数（如 96）时仍可运行（可走 fallback 路径）。

### M3：性能基线与 roofline 分析（优化门禁）

- 实现 flops/bytes 统计：`flops ≈ 4·B·H·N²·d`；bytes 按分块流量公式计算（见第 8 节）。
- 用 `ncu` 采集 `dram__throughput`、`sm__throughput`、L1/L2 命中率、stall 原因。
- 确认瓶颈方向（预期 memory-bound），为后续每步优化提供对照。

**验收**：产出一张 v0/v1 在基准形状上的性能表，判断与带宽上限的差距及主要 stall。

### M4：v2 tile 尺寸调优

- 把 `B_r ∈ {32, 64, 128}`、`B_c ∈ {64, 128, 256}` 做成编译期模板参数；
- 权衡：`B_r` 越大 K/V 重复读越少（省 HBM），但寄存器与 shared 压力越大；
- 保证至少 2 个 CTA 驻留/SM，smem 不超过每 SM 容量；
- 用脚本扫参数，固定最优组合。

**验收**：固定形状下带宽利用率明显提升；参数扫描结果记录到文档。

### M5：v3 向量化加载

- global→shared 全部改为 128-bit（float4）访问，行首 16 字节对齐；
- 尾部不足 4 元素单独处理（或对 `d` 做 padding 约束）；
- 减少标量 load 与地址计算。

**验收**：ncu 显示 `l1tex` 请求数下降、带宽上升，正确性回归通过。

### M6：v4 cp.async 多级流水

- 用 `cp.async`（sm_80+）预取下一 K/V 块，与当前块计算重叠；
- `commit_group` / `wait_group` 实现 2~3 级流水；
- 注意 16 字节对齐约束、尾块处理、同步点收敛到最少。

**验收**：`dram__throughput` 接近 80%+ 峰值，stall 中 memory 等待占比下降。

### M7：v5 Tensor Core

- fp16（或 TF32）执行 `QKᵀ` 与 `P@V` 的 `mma.sync`；
- `m/l/O` 仍为 fp32；P 在进入 MMA 前转 fp16；
- 先用简单 fragment 布局保证正确，再处理 `ldmatrix`、shared memory padding/swizzle 消 bank conflict；
- 若精度敏感，fp32 CUDA-core 路径保留为可选项；
- fp16 下注意先乘 scale 再转换，避免溢出。

**验收**：与 fp32 参考的误差在合理范围（fp16 版本约 `1e-2` 量级），吞吐相对 v4 提升或持平（若已 memory-bound，预期提升有限——这本身是正确结论）。

### M8：v6 causal 算法优化

- causal 时 K 循环只走到对角块，全零上三角整块跳过，对角块单独 mask；
- 长序列下 FLOP 与 HBM 流量接近减半；
- online softmax 的 `l` 只累加有效块，逻辑不变。

**验收**：causal 场景相对非 causal 耗时接近减半；正确性回归通过。

### M9：v7 细节打磨

- `exp2f` + 乘 `log2(e)` 替代 `expf`；
- `1/l` 倒数一次，用乘法归一化；
- `launch_bounds` / `maxrregcount` 调寄存器占用；
- CTA 调度顺序 L2 友好（相邻 Q tile 复用同一批 K/V）；
- 减少 `__syncthreads` 次数，消除 bank conflict。

**验收**：达到目标带宽利用率；性能演进图归档到 `docs/`。

### M10（可选）：v8 进阶

- warp specialization（producer/consumer）；
- 长序列 split-KV / 行切分；
- 对照 FlashAttention-2 的改进思路。

## 7. 正确性验证策略

测试矩阵：

| 维度 | 取值 |
|---|---|
| N | 1, 64, 512, 1024, 2048, 4096 |
| d | 32, 64, 96, 128 |
| B×H | 1×1, 1×8, 4×16 |
| causal | false, true |
| 数据分布 | 均匀随机、大数值（测 softmax 稳定性）、全零行 |

精度标准：fp32 版本对 torch 参考 `atol ≈ rtol ≈ 1e-4`（对 d=96/128 可放宽到 `1e-3`）；fp16 版本 `1e-2` 量级。每个 kernel 版本必须全量回归。

边界用例：

- `N=1`、`d=32` 等最小形状；
- 非 8 倍数 head_dim 的 fallback；
- causal 下对角线 block 的 mask 正确性；
- 输入含 `NaN`/`Inf` 的行为可接受即可（不强制）。

## 8. 基准测试方法

基准形状（覆盖小 batch 长序列与大 batch 短序列）：

```text
(B, H, N, d): (1,1,1024,64)  (1,8,2048,64)  (1,16,4096,128)  (4,16,2048,64)
```

统计口径：

- `flops ≈ 4·B·H·N²·d`（QKᵀ 与 PV 各 `2·N²·d`，causal 按有效块折算）；
- bytes（fp32，非 causal）≈ `(B·H·N·d·4)·3 + (B·H·N²·d·8)/B_r`；
- fp16 数据 bytes 减半；causal 时约再减半；
- 计时：warmup 后多轮取中位数/最小值，用 CUDA events，排除拷贝；
- 与 torch eager attention 同精度对比。

ncu 关键指标：`dram__throughput`、`sm__throughput`、`l1tex__t_sectors`、stall breakdown、occupancy。

## 9. 优化决策依据（roofline 推导）

设分块后算术强度近似为：

```text
intensity ≈ B_r            (fp16 数据)
intensity ≈ B_r / 2        (fp32 数据)
```

| 数据精度 | B_r=64 时强度 | 盈亏平衡点 | 结论 |
|---|---|---|---|
| fp32 + CUDA core | 32 FLOP/B | ~36 FLOP/B | 贴边，接近 memory-bound |
| fp16 + Tensor Core | 64 FLOP/B | ~72 FLOP/B | 仍 memory-bound |

推论：

1. 先优化内存路径（向量化、流水、tile 尺寸），收益最大；
2. Tensor Core 主要价值在减小计算延迟/提高计算余量，不直接解决带宽瓶颈；
3. fp16 同时减半流量并翻倍计算能力，是"内存+计算"双赢的一步；
4. causal 是算法级减半，优先级高于任何微优化。

## 10. 已知坑与注意事项

- **online softmax 顺序**：必须先更新 max 再 rescale 旧状态；
- **mask 时机**：mask 在求 rowmax 之前生效，否则 `-inf` 的 max 污染 softmax；
- **fp16 溢出**：QKᵀ 前先乘 `1/√d`，转换在缩放之后；
- **`cp.async` 对齐**：16 字节对齐，尾块单独处理；
- **bank conflict**：K/V shared tile 按列访问时天然冲突，需要 padding 或 swizzle；
- **同步**：多级流水时 `wait_group` 位置错误会导致读未就绪数据；
- **benchmark 公平性**：同精度对比，排除 H2D/D2H，多轮取稳定值；
- **不要过早使用 `--use_fast_math`**：先保证正确，再作为最后手段并复核精度。

## 11. 参考资料

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022
- 官方实现：`https://github.com/dao-AILab/flash-attention`（研究布局与流水，不照抄）
- Tri Dao 关于 FlashAttention 的博客（算法与复杂度讲解）
- CUTLASS FMHA 相关示例（Tensor Core 布局与流水）
- 本仓库 `src/cuda/gemm/` 系列（v8~v12 的流水与 TC 思路可借鉴）
- NVIDIA `ncu` 用户指南（指标口径）

## 12. 验收总清单

- [x] CMake 集成完成，`USE_CUDA=ON` 可构建测试与 benchmark
- [ ] v0/v1 与 torch 参考对齐（含 causal、边界形状）
- [ ] 性能基线表 + roofline 分析产出
- [ ] v2~v4：带宽利用率逐步逼近 80%+
- [ ] v5：Tensor Core 版本正确且计算不再构成瓶颈
- [ ] v6：causal 场景接近 2 倍收益
- [ ] v7：达到目标性能，演进图归档 `docs/`
- [ ] 所有版本源码保留，测试全量回归通过
