#pragma once
#include "kernel_common.cuh"

namespace v11 {

template <class Config, bool kAligned>
__device__ __forceinline__ void
prefetch_A_reg(float prefA[], const float* __restrict__ A,
               const WarpTile<Config>& warp_tile, int M, int K, int lda,
               int k_next) {
  constexpr int kRowsPerWarp = Config::BLOCK_TILE_M / Config::WARPS;  // 8
  constexpr int kColsPerLane =
      Config::BLOCK_TILE_K / 4;  // 4 lanes per row share BK
  const int lane_id = warp_tile.lane_id;
  const int row = blockIdx.y * Config::BLOCK_TILE_M +
                  warp_tile.warp_id * kRowsPerWarp + (lane_id >> 2);
  // K offset from the float4 group: (lane&3)*kColsPerLane .. +kColsPerLane.
  const int col = k_next + (lane_id & 3) * kColsPerLane;

  if constexpr (kAligned) {
#pragma unroll
    for (int v = 0; v < kColsPerLane / 4; ++v)
      *reinterpret_cast<float4*>(&prefA[4 * v]) =
          __ldcg(reinterpret_cast<const float4*>(&A[row * lda + col + 4 * v]));
  } else {
    const bool pred_m = (row < M);
#pragma unroll
    for (int v = 0; v < kColsPerLane / 4; ++v) {
      const float4 val = (pred_m && (col + 4 * v + 3) < K)
                             ? __ldcg(reinterpret_cast<const float4*>(
                                   &A[row * lda + col + 4 * v]))
                             : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      *reinterpret_cast<float4*>(&prefA[4 * v]) = val;
    }
  }
}

template <class Config>
__device__ __forceinline__ void store_A_reg(float* __restrict__ As,
                                            const float prefA[],
                                            const WarpTile<Config>& warp_tile) {
  constexpr int kRowsPerWarp = Config::BLOCK_TILE_M / Config::WARPS;  // 8
  constexpr int kColsPerLane =
      Config::BLOCK_TILE_K / 4;  // 4 lanes per row share BK
  const int lane_id = warp_tile.lane_id;
  const int row = warp_tile.warp_id * kRowsPerWarp + (lane_id >> 2);
  // K offset from the float4 group: (lane&3)*kColsPerLane .. +kColsPerLane.
  const int k0 = (lane_id & 3) * kColsPerLane;

#pragma unroll
  for (int j = 0; j < kColsPerLane; ++j)
    As[(k0 + j) * Config::SA_STRIDE + row] = prefA[j];
}

template <class Config, bool kAligned>
__device__ __forceinline__ void
prefetch_B_reg(float prefB[], const float* __restrict__ B,
               const WarpTile<Config>& warp_tile, int K, int N, int ldb,
               int k_next) {
  constexpr int kRowsPerWarp =
      Config::BLOCK_TILE_K / Config::WARPS_M;  // 4 (BK=16)
  const int lane_id = warp_tile.lane_id;
  const int global_col = warp_tile.global_col + lane_id;

  if constexpr (kAligned) {
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
      const int global_row = warp_tile.warp_m * kRowsPerWarp + i + k_next;
      prefB[i] = __ldcg(&B[global_row * ldb + global_col]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
      const int global_row = warp_tile.warp_m * kRowsPerWarp + i + k_next;
      prefB[i] = (global_row < K && global_col < N)
                     ? __ldcg(&B[global_row * ldb + global_col])
                     : 0.0f;
    }
  }
}

template <class Config>
__device__ __forceinline__ void store_B_reg(float* __restrict__ Bs,
                                            const float prefB[],
                                            const WarpTile<Config>& warp_tile) {
  constexpr int kRowsPerWarp = Config::BLOCK_TILE_K / Config::WARPS_M;
  const int lane_id = warp_tile.lane_id;
#pragma unroll
  for (int i = 0; i < kRowsPerWarp; ++i) {
    const int block_row = warp_tile.warp_m * kRowsPerWarp + i;
    const int block_col = warp_tile.warp_n * Config::WARP_TILE_N + lane_id;
    Bs[block_row * Config::BLOCK_TILE_N + block_col] = prefB[i];
  }
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

#pragma unroll
  for (int ik = 0; ik < Config::BLOCK_TILE_K; ++ik) {
    float av[Config::THREAD_TILE_M];
    const float* pa = &As[ik * Config::SA_STRIDE + a_row0];
#pragma unroll
    for (int v = 0; v < Config::THREAD_TILE_M / 4; ++v)
      *reinterpret_cast<float4*>(av + 4 * v) =
          *reinterpret_cast<const float4*>(pa + 4 * v);

    float bv[Config::THREAD_TILE_N];
    *reinterpret_cast<float4*>(bv) = *reinterpret_cast<const float4*>(
        &Bs[ik * Config::BLOCK_TILE_N + b_col0]);

#pragma unroll
    for (int i = 0; i < Config::THREAD_TILE_M; ++i)
#pragma unroll
      for (int j = 0; j < Config::THREAD_TILE_N; ++j)
        acc(i, j) += av[i] * bv[j];
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
      float4 out = make_float4(acc(i, 0), acc(i, 1), acc(i, 2), acc(i, 3));
      *reinterpret_cast<float4*>(&C[(row0 + i) * ldc + col0]) = out;
    }
  } else {
    const bool col_ok = (col0 + Config::THREAD_TILE_N - 1) < N;
#pragma unroll
    for (int i = 0; i < Config::THREAD_TILE_M; ++i) {
      if ((row0 + i) < M && col_ok) {
        float4 out = make_float4(acc(i, 0), acc(i, 1), acc(i, 2), acc(i, 3));
        *reinterpret_cast<float4*>(&C[(row0 + i) * ldc + col0]) = out;
      }
    }
  }
}

// __launch_bounds__ toggle for the config sweep: define V11_USE_LAUNCH_BOUNDS
// to 0 before including this header to disable it (default 1 = enabled).
#ifndef V11_USE_LAUNCH_BOUNDS
#define V11_USE_LAUNCH_BOUNDS 1
#endif

#if V11_USE_LAUNCH_BOUNDS
#define V11_LAUNCH_BOUNDS __launch_bounds__(Config::THREADS, 2)
#else
#define V11_LAUNCH_BOUNDS
#endif

template <class Config, bool kAligned>
__global__ void V11_LAUNCH_BOUNDS gemm_v11_kernel(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C, int M,
                                                  int N, int K, int lda,
                                                  int ldb, int ldc) {
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

  float prefA[Config::BLOCK_TILE_K / 4];  // one float4 per lane
  float prefB[Config::BLOCK_TILE_K / Config::WARPS_M];

  const int num_tiles = (K + Config::BLOCK_TILE_K - 1) / Config::BLOCK_TILE_K;

  prefetch_A_reg<Config, kAligned>(prefA, A, warp_tile, M, K, lda, 0);
  prefetch_B_reg<Config, kAligned>(prefB, B, warp_tile, K, N, ldb, 0);

  store_A_reg<Config>(As0, prefA, warp_tile);
  store_B_reg<Config>(Bs0, prefB, warp_tile);
  __syncthreads();

  for (int ph = 0; ph < num_tiles; ++ph) {
    int cur = ph & 1;
    int next = cur ^ 1;

    float* As_cur = cur ? As1 : As0;
    float* Bs_cur = cur ? Bs1 : Bs0;
    float* As_nxt = next ? As1 : As0;
    float* Bs_nxt = next ? Bs1 : Bs0;

    if (ph + 1 < num_tiles) {
      const int k_next = (ph + 1) * Config::BLOCK_TILE_K;
      prefetch_A_reg<Config, kAligned>(prefA, A, warp_tile, M, K, lda, k_next);
      prefetch_B_reg<Config, kAligned>(prefB, B, warp_tile, K, N, ldb, k_next);
    }

    v11::mma<Config>(As_cur, Bs_cur, warp_tile, thread_tile, acc);

    if (ph + 1 < num_tiles) {
      store_A_reg<Config>(As_nxt, prefA, warp_tile);
      store_B_reg<Config>(Bs_nxt, prefB, warp_tile);
      __syncthreads();
    }
  }

  v11::store_C<Config, kAligned>(C, warp_tile, thread_tile, acc, M, N, ldc);
}

}  // namespace v11
