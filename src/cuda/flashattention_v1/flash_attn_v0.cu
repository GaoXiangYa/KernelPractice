#include <cmath>
#include <cstdio>
#include <math.h>

#include "flash_attn.cuh"
#include "flashattention_v1.h"
#include "util.h"

// ---- 32-lane warp reductions (one warp == one query row) ----
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
  return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  return v;
}

// ---- strided global -> shared tile loaders ----
template <typename T, int HEAD_DIM, int BLOCK_M>
__device__ __forceinline__ void load_Q_tile(const FlashAttentionParams& params,
                                            int batch_idx, int head_idx,
                                            int q_start, T* smem_q) {
  constexpr int SMEM_DIM = HEAD_DIM + 1;  // pad 1 float/row to dodge bank conflicts
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  // Strides are element counts; index the tile as a T* base + element offsets.
  const T* base = reinterpret_cast<const T*>(params.q) +
                  batch_idx * params.q_stride_b + head_idx * params.q_stride_h;
  for (int idx = tid; idx < BLOCK_M * HEAD_DIM; idx += blockDim.x * blockDim.y) {
    const int r = idx / HEAD_DIM;
    const int c = idx % HEAD_DIM;
    const int row = q_start + r;
    if (c < params.head_dim && row < params.seq_len_q)
      smem_q[r * SMEM_DIM + c] = base[row * params.q_stride_s + c];
    else
      smem_q[r * SMEM_DIM + c] = T(0);
  }
}

template <typename T, int HEAD_DIM, int BLOCK_N>
__device__ __forceinline__ void load_K_tile(const FlashAttentionParams& params,
                                            int batch_idx, int kv_head_idx,
                                            int k_start, T* smem_k) {
  constexpr int SMEM_DIM = HEAD_DIM + 1;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  // Strides are element counts; index the tile as a T* base + element offsets.
  const T* base = reinterpret_cast<const T*>(params.k) +
                  batch_idx * params.k_stride_b + kv_head_idx * params.k_stride_h;
  for (int idx = tid; idx < BLOCK_N * HEAD_DIM; idx += blockDim.x * blockDim.y) {
    const int r = idx / HEAD_DIM;
    const int c = idx % HEAD_DIM;
    const int row = k_start + r;
    if (c < params.head_dim && row < params.seq_len_k)
      smem_k[r * SMEM_DIM + c] = base[row * params.k_stride_s + c];
    else
      smem_k[r * SMEM_DIM + c] = T(0);
  }
}

template <typename T, int HEAD_DIM, int BLOCK_N>
__device__ __forceinline__ void load_V_tile(const FlashAttentionParams& params,
                                            int batch_idx, int kv_head_idx,
                                            int v_start, T* smem_v) {
  // identical to load_K_tile but uses params.v / params.v_stride_*
  constexpr int SMEM_DIM = HEAD_DIM + 1;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const T* base = reinterpret_cast<const T*>(params.v) +
                  batch_idx * params.v_stride_b + kv_head_idx * params.v_stride_h;
  for (int idx = tid; idx < BLOCK_N * HEAD_DIM; idx += blockDim.x * blockDim.y) {
    const int r = idx / HEAD_DIM;
    const int c = idx % HEAD_DIM;
    const int row = v_start + r;
    if (c < params.head_dim && row < params.seq_len_k)
      smem_v[r * SMEM_DIM + c] = base[row * params.v_stride_s + c];
    else
      smem_v[r * SMEM_DIM + c] = T(0);
  }
}

// ---- v0 kernel: naive fused attention, two-pass softmax per KV block ----
template <typename T, typename AccT, int HEAD_DIM, int BLOCK_M, int BLOCK_N,
          int BLOCK_K, bool CAUSAL>
__global__ void flash_attn_kernel_v0(FlashAttentionParams params) {
  constexpr int SMEM_DIM = HEAD_DIM + 1;

  __shared__ T s_q[BLOCK_M * SMEM_DIM];
  __shared__ T s_k[BLOCK_N * SMEM_DIM];
  __shared__ T s_v[BLOCK_N * SMEM_DIM];

  const int lane = threadIdx.x;  // kv column inside the tile
  const int warp = threadIdx.y;  // query row inside the tile
  const int q_row = blockIdx.x * BLOCK_M + warp;

  const int batch_idx = blockIdx.y / params.num_heads;
  const int head_idx = blockIdx.y % params.num_heads;
  const int kv_head_idx = params.num_kv_heads > 0
                              ? head_idx * params.num_kv_heads / params.num_heads
                              : head_idx;

  const bool row_valid = q_row < params.seq_len_q;

  // Per-lane output accumulator (one warp per query row): lane j holds the
  // partial weighted sum for its own kv columns across all blocks, then a
  // warp shuffle reduction folds the 32 lanes together.
  AccT o_acc[HEAD_DIM] = {};

  load_Q_tile<T, HEAD_DIM, BLOCK_M>(params, batch_idx, head_idx,
                                    blockIdx.x * BLOCK_M, s_q);
  __syncthreads();

  // Pass 1: global row max over the whole KV dimension (register + shuffle).
  // Two-pass softmax normalizes by the row sum, so the max must be global.
  float row_max = -INFINITY;
  for (int k0 = 0; k0 < params.seq_len_k; k0 += BLOCK_N) {
    load_K_tile<T, HEAD_DIM, BLOCK_N>(params, batch_idx, kv_head_idx, k0, s_k);
    __syncthreads();  // K tile ready

    if (row_valid) {
      // S = sum_d Q[d] * K[d] * scale
      AccT acc = AccT(0);
      for (int d0 = 0; d0 < params.head_dim; d0 += BLOCK_K) {
        const int d_end = min(d0 + BLOCK_K, params.head_dim);
        for (int d = d0; d < d_end; ++d)
          acc += (AccT)s_q[warp * SMEM_DIM + d] * (AccT)s_k[lane * SMEM_DIM + d];
      }
      float s_val = (float)(acc * (AccT)params.softmax_scale);
      const int k_col = k0 + lane;
      if (k_col >= params.seq_len_k) s_val = -INFINITY;
      if (CAUSAL && q_row < k_col) s_val = -INFINITY;
      row_max = fmaxf(row_max, s_val);
    }
    __syncthreads();  // s_k overwritten next iteration
  }
  row_max = warp_reduce_max(row_max);

  // Pass 2: accumulate unnormalized exp(s - row_max) * V into per-lane registers.
  float row_sum = 0.0f;
  for (int k0 = 0; k0 < params.seq_len_k; k0 += BLOCK_N) {
    load_K_tile<T, HEAD_DIM, BLOCK_N>(params, batch_idx, kv_head_idx, k0, s_k);
    load_V_tile<T, HEAD_DIM, BLOCK_N>(params, batch_idx, kv_head_idx, k0, s_v);
    __syncthreads();  // K/V tiles ready

    if (row_valid) {
      AccT acc = AccT(0);
      for (int d0 = 0; d0 < params.head_dim; d0 += BLOCK_K) {
        const int d_end = min(d0 + BLOCK_K, params.head_dim);
        for (int d = d0; d < d_end; ++d)
          acc += (AccT)s_q[warp * SMEM_DIM + d] * (AccT)s_k[lane * SMEM_DIM + d];
      }
      float s_val = (float)(acc * (AccT)params.softmax_scale);
      const int k_col = k0 + lane;
      if (k_col >= params.seq_len_k) s_val = -INFINITY;
      if (CAUSAL && q_row < k_col) s_val = -INFINITY;

      if (row_max != -INFINITY) {  // fully masked row contributes nothing (no NaN)
        const float e = expf(s_val - row_max);
        row_sum += e;
        for (int d0 = 0; d0 < params.head_dim; d0 += BLOCK_K) {
          const int d_end = min(d0 + BLOCK_K, params.head_dim);
          for (int d = d0; d < d_end; ++d)
            o_acc[d] += (AccT)(e * s_v[lane * SMEM_DIM + d]);
        }
      }
    }
    __syncthreads();  // s_v/s_k overwritten next iteration
  }
  row_sum = warp_reduce_sum(row_sum);

  if (row_valid) {
    // Strides are element counts; write through a T* base + element offset.
    T* dst = reinterpret_cast<T*>(params.o) + batch_idx * params.o_stride_b +
             head_idx * params.o_stride_h + q_row * params.o_stride_s;
    if (row_sum > 0.0f) {
      // Fold the 32 lanes' partial sums; after the reduction every lane holds
      // the full row value for every d.
      for (int d = 0; d < params.head_dim; ++d) o_acc[d] = warp_reduce_sum(o_acc[d]);
      for (int d = lane; d < params.head_dim; d += BLOCK_N)
        dst[d] = (T)(o_acc[d] / (AccT)row_sum);
    } else {
      for (int d = lane; d < params.head_dim; d += BLOCK_N)
        dst[d] = T(0);
    }
  }
}

namespace {
void launch_v0(const FlashAttentionParams& params) {
  constexpr int BLOCK_M = 8;
  constexpr int BLOCK_N = 32;
  constexpr int BLOCK_K = 32;
  const dim3 block(BLOCK_N, BLOCK_M);  // 256 threads, 8 warps
  const int num_q_blocks = (params.seq_len_q + BLOCK_M - 1) / BLOCK_M;
  const dim3 grid(num_q_blocks, params.batch * params.num_heads);

  const int hd = params.head_dim;
  if (hd > 128) {
    fprintf(stderr, "flash_attn_v0: head_dim %d > 128 not supported\n", hd);
    return;
  }
  // HEAD_DIM instance >= runtime head_dim (96 falls into the 128 instance)
  if (hd <= 32) {
    if (params.causal)
      flash_attn_kernel_v0<float, float, 32, BLOCK_M, BLOCK_N, BLOCK_K, true>
          <<<grid, block>>>(params);
    else
      flash_attn_kernel_v0<float, float, 32, BLOCK_M, BLOCK_N, BLOCK_K, false>
          <<<grid, block>>>(params);
  } else if (hd <= 64) {
    if (params.causal)
      flash_attn_kernel_v0<float, float, 64, BLOCK_M, BLOCK_N, BLOCK_K, true>
          <<<grid, block>>>(params);
    else
      flash_attn_kernel_v0<float, float, 64, BLOCK_M, BLOCK_N, BLOCK_K, false>
          <<<grid, block>>>(params);
  } else {
    if (params.causal)
      flash_attn_kernel_v0<float, float, 128, BLOCK_M, BLOCK_N, BLOCK_K, true>
          <<<grid, block>>>(params);
    else
      flash_attn_kernel_v0<float, float, 128, BLOCK_M, BLOCK_N, BLOCK_K, false>
          <<<grid, block>>>(params);
  }
}
}  // namespace

void flash_attn_v0(const FlashAttentionParams& params) {
  launch_v0(params);
  CHECK_CUDA(cudaGetLastError());
}

void flash_attn_v0(const float* Q, const float* K, const float* V, float* O,
                   int B, int H, int N, int d, bool causal) {
  const int total = B * H * N * d;
  float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
  CHECK_CUDA(cudaMalloc(&dQ, total * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dK, total * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dV, total * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dO, total * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dQ, Q, total * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK, K, total * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV, V, total * sizeof(float), cudaMemcpyHostToDevice));

  FlashAttentionParams p = make_flash_attn_params(
      dQ, dK, dV, dO, B, H, N, N, d, 1.0f / std::sqrt((float)d), causal);
  flash_attn_v0(p);

  CHECK_CUDA(cudaMemcpy(O, dO, total * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(dQ));
  CHECK_CUDA(cudaFree(dK));
  CHECK_CUDA(cudaFree(dV));
  CHECK_CUDA(cudaFree(dO));
}
