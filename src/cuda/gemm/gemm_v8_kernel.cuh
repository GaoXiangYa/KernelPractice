#pragma once
#include "kernel_common.cuh"

// ============================================================================
// gemm_v8: warp-level tiled GEMM built on the Config/WarpTile/ThreadTile
// framework.  All indexing goes through Config constants / tile structs.
//
//   As: A tile stored TRANSPOSED,  As[k][row], stride = SA_STRIDE
//   Bs: B tile stored directly,    Bs[k][col], stride = BLOCK_TILE_N
// ============================================================================

// ---- load A tile: K = lane_id (32), row = warp_id * ROWS_PER_WARP + i ----
template <class Config>
__device__ __forceinline__ void
load_A(float* __restrict__ As, const float* __restrict__ A,
       const WarpTile<Config>& warp_tile, int M, int K, int lda, int k) {
  constexpr int kRowsPerWarp = Config::BLOCK_TILE_M / Config::WARPS;  // 8
  const int lane_id = warp_tile.lane_id;      // 0..31 = BK
  const int gk = k + lane_id;
  const bool k_ok = (gk < K);

#pragma unroll
  for (int i = 0; i < kRowsPerWarp; ++i) {
    const int block_row = warp_tile.warp_id * kRowsPerWarp + i;  // 0..BM-1
    const int global_row = blockIdx.y * Config::BLOCK_TILE_M + block_row;
    const bool m_ok = (global_row < M);
    As[lane_id * Config::SA_STRIDE + block_row] =
        (k_ok && m_ok) ? A[global_row * lda + gk] : 0.0f;
  }
}

// ---- load B tile: K = warp_m * ROWS_PER_WARPM + i, col = warp_n*WN+lane ----
template <class Config>
__device__ __forceinline__ void
load_B(float* __restrict__ Bs, const float* __restrict__ B,
       const WarpTile<Config>& warp_tile, int K, int N, int ldb, int k) {
  constexpr int kRowsPerWarpM = Config::BLOCK_TILE_K / Config::WARPS_M;  // 8
  const int lane_id = warp_tile.lane_id;

#pragma unroll
  for (int i = 0; i < kRowsPerWarpM; ++i) {
    const int k_row = warp_tile.warp_m * kRowsPerWarpM + i;  // 0..BK-1
    const int block_col = warp_tile.warp_n * Config::WARP_TILE_N + lane_id;  // 0..BN-1
    const int global_col = warp_tile.global_col + lane_id;
    const int gk = k + k_row;
    Bs[k_row * Config::BLOCK_TILE_N + block_col] =
        (gk < K && global_col < N) ? B[gk * ldb + global_col] : 0.0f;
  }
}

// ---- mma: av = As[ik][thread 8 rows] (2×float4), bv = Bs[ik][thread 4 cols] ----
template <class Config>
__device__ __forceinline__ void
mma(const float* __restrict__ As, const float* __restrict__ Bs,
    const WarpTile<Config>& warp_tile, const ThreadTile<Config>& thread_tile,
    RegisterTile<float, Config::THREAD_TILE_M, Config::THREAD_TILE_N>& acc) {
  const int a_row0 = warp_tile.warp_m * Config::WARP_TILE_M
                   + thread_tile.thread_m * Config::THREAD_TILE_M;
  const int b_col0 = warp_tile.warp_n * Config::WARP_TILE_N
                   + thread_tile.thread_n * Config::THREAD_TILE_N;

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

// ---- store C: per row 4 consecutive cols → float4 coalesced store ----
template <class Config>
__device__ __forceinline__ void
store_C(float* __restrict__ C, const WarpTile<Config>& warp_tile,
        const ThreadTile<Config>& thread_tile,
        const RegisterTile<float, Config::THREAD_TILE_M, Config::THREAD_TILE_N>&
            acc,
        int M, int N, int ldc) {
  const int row0 = thread_tile.global_row;
  const int col0 = thread_tile.global_col;
  const bool col_ok = (col0 + Config::THREAD_TILE_N - 1) < N;

#pragma unroll
  for (int i = 0; i < Config::THREAD_TILE_M; ++i) {
    if ((row0 + i) < M && col_ok) {
      float4 out = make_float4(acc(i, 0), acc(i, 1), acc(i, 2), acc(i, 3));
      *reinterpret_cast<float4*>(&C[(row0 + i) * ldc + col0]) = out;
    }
  }
}

template <class Config>
__global__ void gemm_v8_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K,
                               int lda, int ldb, int ldc) {
  WarpTile<Config> warp_tile;
  ThreadTile<Config> thread_tile = make_thread_tile(warp_tile);
  RegisterTile<float, Config::THREAD_TILE_M, Config::THREAD_TILE_N> acc;
  acc.clear();

  __shared__ float As[Config::BLOCK_TILE_K * Config::SA_STRIDE];
  __shared__ float Bs[Config::BLOCK_TILE_K * Config::BLOCK_TILE_N];

  for (int k = 0; k < K; k += Config::BLOCK_TILE_K) {
    load_A<Config>(As, A, warp_tile, M, K, lda, k);
    load_B<Config>(Bs, B, warp_tile, K, N, ldb, k);
    __syncthreads();

    mma<Config>(As, Bs, warp_tile, thread_tile, acc);
    __syncthreads();
  }

  store_C<Config>(C, warp_tile, thread_tile, acc, M, N, ldc);
}
