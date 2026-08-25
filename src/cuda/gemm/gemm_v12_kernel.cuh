#pragma once
#include "kernel_common.cuh"

namespace v12 {

template <class Config, bool kAligned>
__device__ __forceinline__ void
prefetch_A_reg(float prefA[], const float* __restrict__ A,
               const WarpTile<Config>& warp_tile, int M, int K, int lda,
               int k_next) {
  constexpr int kRowsPerWarp = Config::BLOCK_TILE_M / Config::WARPS;  // 16
  constexpr int kColsPerLane =
      Config::BLOCK_TILE_K * Config::BLOCK_TILE_M / (32 * Config::WARPS);  // 8
  constexpr int kLanesPerRow = Config::BLOCK_TILE_K / kColsPerLane;        // 2
  const int lane_id = warp_tile.lane_id;
  const int row = blockIdx.y * Config::BLOCK_TILE_M +
                  warp_tile.warp_id * kRowsPerWarp + lane_id / kLanesPerRow;
  const int col = k_next + (lane_id % kLanesPerRow) * kColsPerLane;

  if constexpr (kAligned) {
#pragma unroll
    for (int j = 0; j < kColsPerLane; j += 4)
      *reinterpret_cast<float4*>(&prefA[j]) =
          __ldcg(reinterpret_cast<const float4*>(&A[row * lda + col + j]));
  } else {
    const bool pred_m = (row < M);
#pragma unroll
    for (int j = 0; j < kColsPerLane; j += 4) {
      const float4 v =
          (pred_m && (col + j + 3) < K)
              ? __ldcg(reinterpret_cast<const float4*>(&A[row * lda + col + j]))
              : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      *reinterpret_cast<float4*>(&prefA[j]) = v;
    }
  }
}

template <class Config>
__device__ __forceinline__ void store_A_reg(float* __restrict__ As,
                                            const float prefA[],
                                            const WarpTile<Config>& warp_tile) {
  constexpr int kRowsPerWarp = Config::BLOCK_TILE_M / Config::WARPS;  // 16
  constexpr int kColsPerLane =
      Config::BLOCK_TILE_K * Config::BLOCK_TILE_M / (32 * Config::WARPS);  // 8
  constexpr int kLanesPerRow = Config::BLOCK_TILE_K / kColsPerLane;        // 2
  const int lane_id = warp_tile.lane_id;
  const int row = warp_tile.warp_id * kRowsPerWarp + lane_id / kLanesPerRow;
  const int k0 = (lane_id % kLanesPerRow) * kColsPerLane;

#pragma unroll
  for (int j = 0; j < kColsPerLane; ++j)
    As[(k0 + j) * Config::SA_STRIDE + row] = prefA[j];
}

template <class Config, bool kAligned>
__device__ __forceinline__ void
prefetch_B_reg(float prefB[], const float* __restrict__ B,
               const WarpTile<Config>& warp_tile, int K, int N, int ldb,
               int k_next) {
  constexpr int kRowsPerWarp = Config::BLOCK_TILE_K / Config::WARPS_M;   // 4
  constexpr int kColsPerLane = Config::WARP_TILE_N * kRowsPerWarp / 32;  // 8
  constexpr int kLanesPerRow = 32 / kRowsPerWarp;                        // 8
  const int lane_id = warp_tile.lane_id;
  const int block_row =
      warp_tile.warp_m * kRowsPerWarp + lane_id / kLanesPerRow;
  const int global_row = block_row + k_next;
  const int global_col =
      warp_tile.global_col + (lane_id % kLanesPerRow) * kColsPerLane;

  if constexpr (kAligned) {
#pragma unroll
    for (int j = 0; j < kColsPerLane; j += 4)
      *reinterpret_cast<float4*>(&prefB[j]) =
          __ldcg(reinterpret_cast<const float4*>(
              &B[global_row * ldb + global_col + j]));
  } else {
    const bool pred_k = (global_row < K);
#pragma unroll
    for (int j = 0; j < kColsPerLane; j += 4) {
      const float4 v = (pred_k && (global_col + j + 3) < N)
                           ? __ldcg(reinterpret_cast<const float4*>(
                                 &B[global_row * ldb + global_col + j]))
                           : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      *reinterpret_cast<float4*>(&prefB[j]) = v;
    }
  }
}

template <class Config>
__device__ __forceinline__ void store_B_reg(float* __restrict__ Bs,
                                            const float prefB[],
                                            const WarpTile<Config>& warp_tile) {
  constexpr int kRowsPerWarp = Config::BLOCK_TILE_K / Config::WARPS_M;   // 4
  constexpr int kColsPerLane = Config::WARP_TILE_N * kRowsPerWarp / 32;  // 8
  constexpr int kLanesPerRow = 32 / kRowsPerWarp;                        // 8
  const int lane_id = warp_tile.lane_id;
  const int block_row =
      warp_tile.warp_m * kRowsPerWarp + lane_id / kLanesPerRow;
  const int block_col = warp_tile.warp_n * Config::WARP_TILE_N +
                        (lane_id % kLanesPerRow) * kColsPerLane;

#pragma unroll
  for (int j = 0; j < kColsPerLane; j += 4)
    *reinterpret_cast<float4*>(
        &Bs[block_row * Config::BLOCK_TILE_N + block_col + j]) =
        *reinterpret_cast<const float4*>(&prefB[j]);
}

template <class Config>
__device__ __forceinline__ static void
mma(const float* __restrict__ As, const float* __restrict__ Bs,
    const WarpTile<Config>& warp_tile, const ThreadTile<Config>& thread_tile,
    RegisterTile<float, Config::THREAD_TILE_M, Config::THREAD_TILE_N>& acc) {
  const int a_row0 = warp_tile.warp_m * Config::WARP_TILE_M +
                     thread_tile.thread_m * Config::THREAD_TILE_M;
  const int b_col0 = warp_tile.warp_n * Config::WARP_TILE_N +
                     thread_tile.thread_n * Config::THREAD_TILE_N;

  float fa[2][Config::THREAD_TILE_M];
  float fb[2][Config::THREAD_TILE_N];

  {  // prologue: load ik=0 fragments
    const float* pa = &As[a_row0];
#pragma unroll
    for (int v = 0; v < Config::THREAD_TILE_M / 4; ++v)
      *reinterpret_cast<float4*>(&fa[0][4 * v]) =
          *reinterpret_cast<const float4*>(pa + 4 * v);
    const float* pb = &Bs[b_col0];
#pragma unroll
    for (int v = 0; v < Config::THREAD_TILE_N / 4; ++v)
      *reinterpret_cast<float4*>(&fb[0][4 * v]) =
          *reinterpret_cast<const float4*>(pb + 4 * v);
  }

#pragma unroll
  for (int ik = 0; ik < Config::BLOCK_TILE_K; ++ik) {
    const int cur = ik & 1;
    const int nxt = cur ^ 1;
    if (ik + 1 < Config::BLOCK_TILE_K) {
      const float* pa = &As[(ik + 1) * Config::SA_STRIDE + a_row0];
#pragma unroll
      for (int v = 0; v < Config::THREAD_TILE_M / 4; ++v)
        *reinterpret_cast<float4*>(&fa[nxt][4 * v]) =
            *reinterpret_cast<const float4*>(pa + 4 * v);
      const float* pb = &Bs[(ik + 1) * Config::BLOCK_TILE_N + b_col0];
#pragma unroll
      for (int v = 0; v < Config::THREAD_TILE_N / 4; ++v)
        *reinterpret_cast<float4*>(&fb[nxt][4 * v]) =
            *reinterpret_cast<const float4*>(pb + 4 * v);
    }
#pragma unroll
    for (int i = 0; i < Config::THREAD_TILE_M; ++i)
#pragma unroll
      for (int j = 0; j < Config::THREAD_TILE_N; ++j)
        acc(i, j) += fa[cur][i] * fb[cur][j];
  }
}

template <class Config, bool kAligned>
__device__ __forceinline__ static void
store_C(float* __restrict__ C, const WarpTile<Config>& warp_tile,
        const ThreadTile<Config>& thread_tile,
        const RegisterTile<float, Config::THREAD_TILE_M, Config::THREAD_TILE_N>&
            acc,
        int M, int N, int ldc) {
  const int row0 = thread_tile.global_row;
  const int col0 = thread_tile.global_col;

  if constexpr (kAligned) {
#pragma unroll
    for (int i = 0; i < Config::THREAD_TILE_M; ++i) {
      *reinterpret_cast<float4*>(&C[(row0 + i) * ldc + col0]) =
          make_float4(acc(i, 0), acc(i, 1), acc(i, 2), acc(i, 3));
      *reinterpret_cast<float4*>(&C[(row0 + i) * ldc + col0 + 4]) =
          make_float4(acc(i, 4), acc(i, 5), acc(i, 6), acc(i, 7));
    }
  } else {
    const bool col_ok = (col0 + Config::THREAD_TILE_N - 1) < N;
#pragma unroll
    for (int i = 0; i < Config::THREAD_TILE_M; ++i) {
      if ((row0 + i) < M && col_ok) {
        *reinterpret_cast<float4*>(&C[(row0 + i) * ldc + col0]) =
            make_float4(acc(i, 0), acc(i, 1), acc(i, 2), acc(i, 3));
        *reinterpret_cast<float4*>(&C[(row0 + i) * ldc + col0 + 4]) =
            make_float4(acc(i, 4), acc(i, 5), acc(i, 6), acc(i, 7));
      }
    }
  }
}

template <class Config, bool kAligned>
__global__ void gemm_v12_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K,
                                int lda, int ldb, int ldc) {
  WarpTile<Config> warp_tile;
  ThreadTile<Config> thread_tile = make_thread_tile(warp_tile);
  RegisterTile<float, Config::THREAD_TILE_M, Config::THREAD_TILE_N> acc;

  acc.clear();

  extern __shared__ float smem[];
  constexpr int kOffsetA = Config::BLOCK_TILE_K * Config::SA_STRIDE;
  constexpr int kOffsetB = Config::BLOCK_TILE_K * Config::BLOCK_TILE_N;

  float* As0 = smem;
  float* As1 = smem + kOffsetA;
  float* Bs0 = smem + 2 * kOffsetA;
  float* Bs1 = smem + 2 * kOffsetA + kOffsetB;

  float
      prefA[Config::BLOCK_TILE_K * Config::BLOCK_TILE_M / (32 * Config::WARPS)];
  float prefB[Config::WARP_TILE_N * Config::BLOCK_TILE_K /
              (32 * Config::WARPS_M)];

  const int num_tiles = (K + Config::BLOCK_TILE_K - 1) / Config::BLOCK_TILE_K;

  prefetch_A_reg<Config, kAligned>(prefA, A, warp_tile, M, K, lda, 0);
  prefetch_B_reg<Config, kAligned>(prefB, B, warp_tile, K, N, ldb, 0);
  store_A_reg<Config>(As0, prefA, warp_tile);
  store_B_reg<Config>(Bs0, prefB, warp_tile);
  __syncthreads();

  int cur = 0;
  for (int ph = 0; ph < num_tiles - 1; ++ph) {
    const int nxt = cur ^ 1;
    const int k_next = (ph + 1) * Config::BLOCK_TILE_K;
    float* As_cur = cur ? As1 : As0;
    float* Bs_cur = cur ? Bs1 : Bs0;
    float* As_nxt = nxt ? As1 : As0;
    float* Bs_nxt = nxt ? Bs1 : Bs0;

    prefetch_A_reg<Config, kAligned>(prefA, A, warp_tile, M, K, lda, k_next);
    prefetch_B_reg<Config, kAligned>(prefB, B, warp_tile, K, N, ldb, k_next);

    v12::mma<Config>(As_cur, Bs_cur, warp_tile, thread_tile, acc);

    store_A_reg<Config>(As_nxt, prefA, warp_tile);
    store_B_reg<Config>(Bs_nxt, prefB, warp_tile);
    __syncthreads();
    cur = nxt;
  }
  v12::mma<Config>(cur ? As1 : As0, cur ? Bs1 : Bs0, warp_tile, thread_tile,
                   acc);

  v12::store_C<Config, kAligned>(C, warp_tile, thread_tile, acc, M, N, ldc);
}

}  // namespace v12
