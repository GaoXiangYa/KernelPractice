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

__kernel void gemm_v9_kernel(__global const float* restrict A, __global const float* restrict B,
                             __global float* restrict C, const int M, const int N,
                             const int K, float alpha, float beta) {
  const int local_col = get_local_id(0);
  const int local_row = get_local_id(1);

  const int global_col_base = get_group_id(0) * TILE_N * REG_TILE_N + local_col;
  const int global_row_base = get_group_id(1) * TILE_M * REG_TILE_M + local_row;

  __local float shmem_a[2][REG_TILE_M][TILE_M * (TILE_K + 1)];
  __local float shmem_b[2][REG_TILE_N][TILE_K * (TILE_N + 1)];

  // register double buffer
  float frag_a[2][REG_TILE_M];
  float frag_b[2][REG_TILE_N];
  float frag_c[REG_TILE_M * REG_TILE_N] = {0.0f};

  int shmem_cur = 0;
  int shmem_next = 1;

  int num_tiles = (K + TILE_K - 1) / TILE_K;

  // preload tile 0
  {
    int base = 0;
#pragma unroll
    for (int m = 0; m < REG_TILE_M; ++m) {
      int global_row = global_row_base + m * TILE_M;
      int global_col = base + local_col;

      if (global_row < M && global_col < K) {
        shmem_a[shmem_cur][m][local_row * (TILE_K + 1) + local_col] =
            A[global_row * K + global_col];
      } else {
        shmem_a[shmem_cur][m][local_row * (TILE_K + 1) + local_col] = 0.0f;
      }
    }

#pragma unroll
    for (int n = 0; n < REG_TILE_N; ++n) {
      int global_row = base + local_row;
      int global_col = global_col_base + n * TILE_N;

      if (global_row < K && global_col < N) {
        shmem_b[shmem_cur][n][local_row * (TILE_N + 1) + local_col] =
            B[global_row * N + global_col];
      } else {
        shmem_b[shmem_cur][n][local_row * (TILE_N + 1) + local_col] = 0.0f;
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // tile pipeline
  for (int ph = 0; ph < num_tiles; ++ph) {
    // preload next tile
    if (ph + 1 < num_tiles) {
      const int base = (ph + 1) * TILE_K;

#pragma unroll
      for (int m = 0; m < REG_TILE_M; ++m) {
        int global_row = global_row_base + m * TILE_M;
        int global_col = base + local_col;

        if (global_row < M && global_col < K) {
          shmem_a[shmem_next][m][local_row * (TILE_K + 1) + local_col] =
              A[global_row * K + global_col];
        } else {
          shmem_a[shmem_next][m][local_row * (TILE_K + 1) + local_col] = 0.0f;
        }
      }

#pragma unroll
      for (int n = 0; n < REG_TILE_N; ++n) {
        int global_row = base + local_row;
        int global_col = global_col_base + n * TILE_N;

        if (global_row < K && global_col < N) {
          shmem_b[shmem_next][n][local_row * (TILE_N + 1) + local_col] =
              B[global_row * N + global_col];
        } else {
          shmem_b[shmem_next][n][local_row * (TILE_N + 1) + local_col] = 0.0f;
        }
      }
    }
    // barrier(CLK_LOCAL_MEM_FENCE);

    // compute
    int reg_cur = 0, reg_next = 1;

    // preload k=0
#pragma unroll
    for (int m = 0; m < REG_TILE_M; ++m) {
      frag_a[reg_cur][m] = shmem_a[shmem_cur][m][local_row * (TILE_K + 1) + 0];
    }

#pragma unroll
    for (int n = 0; n < REG_TILE_N; ++n) {
      frag_b[reg_cur][n] = shmem_b[shmem_cur][n][0 * (TILE_N + 1) + local_col];
    }

    for (int k = 0; k < TILE_K - 1; ++k) {
#pragma unroll
      for (int m = 0; m < REG_TILE_M; ++m) {
        frag_a[reg_next][m] = shmem_a[shmem_cur][m][local_row * (TILE_K + 1) + k + 1];
      }

#pragma unroll
      for (int n = 0; n < REG_TILE_N; ++n) {
        frag_b[reg_next][n] = shmem_b[shmem_cur][n][(k + 1) * (TILE_N + 1) + local_col];
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

    // last k
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

  // =========================
  // store C
  // =========================
#pragma unroll
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