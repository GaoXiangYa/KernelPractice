#define COARSE_FACTOR 4

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

#define REG_TILE_M 4
#define REG_TILE_N 4

// A[M, K] * B[K, N] = C[M, N]
// shared memory + thread coarse + bank confict + reigster tiling
// register tiling
// A(i~i + REG_TILE_M, k) * B(k, j~j + REG_TILE_N)
// C(i, j),     C(i, j + 1),     C(i, j + 2),     C(i, j + 3)
// C(i + 1, j), C(i + 1, j + 1), C(i + 1, j + 2), C(i + 1, j + 3)
// C(i + 2, j), C(i + 2, j + 1), C(i + 2, j + 2), C(i + 2, j + 3)
// C(i + 3, j), C(i + 3, j + 1), C(i + 3, j + 2), C(i + 3, j + 3)

__kernel void gemm_v7_kernel(__global const float* A, __global const float* B,
                             __global float* C, const int M, const int N,
                             const int K, float alpha, float beta) {
  const int local_row = get_local_id(1);
  const int local_col = get_local_id(0);

  const int global_row_base = TILE_M * get_group_id(1) * REG_TILE_M + local_row;
  const int global_col_base = TILE_N * get_group_id(0) * REG_TILE_N + local_col;

  __local float shmem_a[REG_TILE_M][TILE_M * (TILE_K + 1)];
  __local float shmem_b[REG_TILE_N][TILE_K * (TILE_N + 1)];

  // float sum[COARSE_FACTOR] = {0.0f};
  float sum[REG_TILE_M * REG_TILE_N] = {0.0f};
  float frag_a[REG_TILE_M] = {0.0f};
  float frag_b[REG_TILE_N] = {0.0f};

  const int num_tiles = (K + TILE_K - 1) / TILE_K;
  for (int ph = 0; ph < num_tiles; ++ph) {
    const int base = ph * TILE_K;

#pragma unroll
    // load A(i~i + REG_TILE_M, k) in shared memory
    for (int m = 0; m < REG_TILE_M; ++m) {
      const int global_row = global_row_base + m * TILE_M;
      const int tiled_col = base + local_col;
      if (global_row < M && tiled_col < K) {
        shmem_a[m][local_row * (TILE_K + 1) + local_col] = A[global_row * K + tiled_col];
      } else {
        shmem_a[m][local_row * (TILE_K + 1) + local_col] = 0.0f;
      }
    }

#pragma unroll
    // load B(k, j ~ j + REG_TILE_N) in shared memory
    for (int n = 0; n < REG_TILE_N; ++ n) {
      const int tiled_row = base + local_row;
      const int global_col = global_col_base + n * TILE_N;
      if (tiled_row < K && global_col < N) {
        shmem_b[n][local_row * (TILE_N + 1) + local_col] = B[tiled_row * N + global_col];
      } else {
        shmem_b[n][local_row * (TILE_N + 1) + local_col] = 0.0f;
      }
    }
    

  }
  //   for (int ph = 0; ph < num_tiles; ++ph) {
  //     const int base = ph * TILE_K;

  //     // load matrix A
  //     const int tiled_col = base + local_col;
  //     if (global_row < M && tiled_col < K) {
  //       shmem_a[local_row * (TILE_K + 1) + local_col] =
  //           A[global_row * K + tiled_col];
  //     } else {
  //       shmem_a[local_row * (TILE_K + 1) + local_col] = 0.0f;
  //     }

  // #pragma unroll
  //     // load matrix B
  //     for (int c = 0; c < COARSE_FACTOR; ++c) {
  //       const int tile_row = base + local_row;
  //       const int global_col = global_col_base + c * TILE_N;
  //       if (tile_row < K && global_col < N) {
  //         shmem_b[c][local_row * (TILE_N + 1) + local_col] =
  //             B[tile_row * N + global_col];
  //       } else {
  //         shmem_b[c][local_row * (TILE_N + 1) + local_col] = 0.0f;
  //       }
  //     }

  //     // sync
  //     barrier(CLK_LOCAL_MEM_FENCE);

  //     float a_val;
  //     for (int k = 0; k < TILE_K; ++k) {
  //       a_val = shmem_a[local_row * (TILE_K + 1) + k];
  // #pragma unroll
  //       for (int c = 0; c < COARSE_FACTOR; ++c) {
  //         sum[c] += a_val * shmem_b[c][k * (TILE_N + 1) + local_col];
  //       }
  //     }
  //     // sync
  //     barrier(CLK_LOCAL_MEM_FENCE);
  //   }

#pragma unroll
  for (int c = 0; c < COARSE_FACTOR; ++c) {
    int global_col = global_col_base + c * TILE_N;
    if (global_row < M && global_col < N) {
      int idx = global_row * N + global_col;
      float old = C[idx];
      C[idx] = alpha * sum[c] + beta * old;
    }
  }
}
