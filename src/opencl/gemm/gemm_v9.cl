#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

#define VEC_WIDTH 4

#define REG_TILE_M 4
#define REG_TILE_N 4

// A[M, K] * B[K, N] = C[M, N]
// shared memory + thread coarse + bank confict + reigster tiling + double
// buffer

// register tiling
// A(i~i + REG_TILE_M, k) * B(k, j~j + REG_TILE_N)
// C(i, j),     C(i, j + 1),     C(i, j + 2),     C(i, j + 3)
// C(i + 1, j), C(i + 1, j + 1), C(i + 1, j + 2), C(i + 1, j + 3)
// C(i + 2, j), C(i + 2, j + 1), C(i + 2, j + 2), C(i + 2, j + 3)
// C(i + 3, j), C(i + 3, j + 1), C(i + 3, j + 2), C(i + 3, j + 3)

__kernel void gemm_v9_kernel(__global const float* A, __global const float* B,
                             __global float* C, const int M, const int N,
                             const int K, float alpha, float beta) {
  const int local_col = get_local_id(0);
  const int local_row = get_local_id(1);

  const int global_col_base = get_group_id(0) * TILE_N * REG_TILE_N + local_col;
  const int global_row_base = get_group_id(1) * TILE_M * REG_TILE_M + local_row;

  // shared memory double buffer
  __local shmem_a[2][REG_TILE_M * TILE_M * TILE_K];
  __local shmem_b[2][REG_TILE_N * TILE_K * TILE_N];

  int a_idx = local_row * TILE_K + local_col;
  int b_idx = local_row * TILE_N + local_col;

  // register double buffer
  float frag_a[2][REG_TILE_M];
  float frag_b[2][REG_TILE_N];
  float frag_c[REG_TILE_M * REG_TILE_N] = {0.0f};

  int shmem_cur = 0;
  int shmem_next = 1;

  int num_tiles = (K + TILE_K - 1) / TILE_K;
  int ph = 0;
  int base = ph * TILE_K;
  int tiled_col = base + local_col;
  int tiled_row = base + local_row;
  int a_stride = TILE_M * TILE_K;
  int b_stride = TILE_K * TILE_N;

  // preload tile 0
  {
#pragma unroll
    // load A from global memory to shared memory
    for (int m = 0; m < REG_TILE_M; ++m) {
      int global_row = global_row_base + m * TILE_M;
      if (global_row < M && tiled_col < K) {
        shmem_a[shmem_cur][m * a_stride + a_idx] =
            A[global_row * K + tiled_col];
      } else {
        shmem_a[shmem_cur][m * a_stride + a_idx] = 0.0f;
      }
    }

#pragma unroll
    // load B from global memory to shared memory
    for (int n = 0; n < REG_TILE_N; ++n) {
      int global_col = global_col_base + n * TILE_N;
      if (tiled_row < K && global_col < N) {
        shmem_b[shmem_cur][n * b_stride + b_idx] =
            B[tiled_row * N + global_col];
      } else {
        shmem_b[shmem_cur][n * b_stride + b_idx] = 0.0f;
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // tile pileline
  for (; ph < num_tiles; ++ph) {
    // async load next tile from global memory
    if (ph + 1 < num_tiles) {
      base = (ph + 1) * TILE_K;
      tiled_col = base + local_col;
      tiled_row = base + local_row;

#pragma unroll
      // load A from global memory to shared memory
      for (int m = 0; m < REG_TILE_M; ++m) {
        int global_row = global_row_base + m * TILE_M;
        if (global_row < M && tiled_col < K) {
          shmem_a[shmem_next][m * a_stride + a_idx] =
              A[global_row * K + tiled_col];
        } else {
          shmem_a[shmem_next][m * a_stride + a_idx] = 0.0f;
        }
      }

#pragma unroll
      // load B from global memory to shared memory
      for (int n = 0; n < REG_TILE_N; ++n) {
        int global_col = global_col_base + n * TILE_N;
        if (tiled_row < K && global_col < N) {
          shmem_b[shmem_next][n * b_stride + b_idx] =
              B[tiled_row * N + global_col];
        } else {
          shmem_b[shmem_next][n * b_stride + b_idx] = 0.0f;
        }
      }
    }

    // register pipeline
    // preload k = 0
    int reg_cur = 0;
    int reg_next = 1;
    int k = 0;
#pragma unroll
    // load A from shared memory to register
    for (int m = 0; m < REG_TILE_M; ++m) {
      frag_a[reg_cur][m] =
          shmem_a[shmem_cur][m * a_stride + local_row * TILE_K + k];
    }

#pragma unroll
    // load B from shared memory to register
    for (int n = 0; n < REG_TILE_N; ++n) {
      frag_b[reg_cur][n] =
          shmem_b[shmem_cur][n * b_stride + k * TILE_N + local_col];
    }
    for (; k < TILE_K - 1; ++k) {
      // reg_next = reg_cur ^ 1;
      // preload next fragment from shared memory to register
#pragma unroll
      // load A from shared memory to register
      for (int m = 0; m < REG_TILE_M; ++m) {
        frag_a[reg_next][m] =
            shmem_a[shmem_cur][m * a_stride + local_row * TILE_K + (k + 1)];
      }

#pragma unroll
      // load B from shared memory to register
      for (int n = 0; n < REG_TILE_N; ++n) {
        frag_b[reg_next][n] =
            shmem_b[shmem_cur][n * b_stride + (k + 1) * TILE_N + local_col];
      }

#pragma unroll
      for (int m = 0; m < REG_TILE_M; ++m) {
#pragma unroll
        for (int n = 0; n < REG_TILE_N; ++n) {
          frag_c[m * REG_TILE_N + n] += frag_a[reg_cur][m] * frag_b[reg_cur][n];
        }
      }
      reg_cur ^= 1;
      reg_next ^= 1;
    }
    // compute last k
#pragma unroll
    for (int m = 0; m < REG_TILE_M; ++m) {
#pragma unroll
      for (int n = 0; n < REG_TILE_N; ++n) {
        frag_c[m * REG_TILE_N + n] += frag_a[reg_cur][m] * frag_b[reg_cur][n];
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    shmem_cur ^= 1;
    shmem_next ^= 1;
  }

  // barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
  // store C(i, j) in global memory
  for (int m = 0; m < REG_TILE_M; ++m) {
    const int global_row = global_row_base + m * TILE_M;
    if (global_row < M) {
#pragma unroll
      for (int n = 0; n < REG_TILE_N; ++n) {
        const int global_col = global_col_base + n * TILE_N;
        if (global_col < N) {
          const int idx = global_row * N + global_col;
          C[idx] = alpha * frag_c[m * REG_TILE_N + n] + beta * C[idx];
        }
      }
    }
  }
}