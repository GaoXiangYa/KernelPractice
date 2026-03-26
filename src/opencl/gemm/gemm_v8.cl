#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

#define VEC_WIDTH 4

#define REG_TILE_M 4
#define REG_TILE_N 4

// A[M, K] * B[K, N] = C[M, N]
// shared memory + thread coarse + bank confict + reigster tiling +
// vectorization load + transpose matrix

// register tiling
// A(i~i + REG_TILE_M, k) * B(k, j~j + REG_TILE_N)
// C(i, j),     C(i, j + 1),     C(i, j + 2),     C(i, j + 3)
// C(i + 1, j), C(i + 1, j + 1), C(i + 1, j + 2), C(i + 1, j + 3)
// C(i + 2, j), C(i + 2, j + 1), C(i + 2, j + 2), C(i + 2, j + 3)
// C(i + 3, j), C(i + 3, j + 1), C(i + 3, j + 2), C(i + 3, j + 3)

__kernel void gemm_v8_kernel(__global const float* A, __global const float* B,
                             __global float* C, const int M, const int N,
                             const int K, float alpha, float beta) {
  const int local_row = get_local_id(1);
  const int local_col = get_local_id(0);

  const int global_row_base = TILE_M * get_group_id(1) * REG_TILE_M + local_row;
  const int global_col_base = TILE_N * get_group_id(0) * REG_TILE_N + local_col;

  // transpose matrix a
  __local float shmem_a[REG_TILE_M][TILE_K * (TILE_M + 1)];
  __local float shmem_b[REG_TILE_N][TILE_K * (TILE_N + 1)];

  float sum[REG_TILE_M * REG_TILE_N] = {0.0f};
  float frag_a[REG_TILE_M] = {0.0f};
  float frag_b[REG_TILE_N] = {0.0f};

  const int num_tiles = (K + TILE_K - 1) / TILE_K;
  int vec_col = local_col * VEC_WIDTH;

  for (int ph = 0; ph < num_tiles; ++ph) {
    const int base = ph * TILE_K;

#pragma unroll
    for (int m = 0; m < REG_TILE_M; ++m) {
      const int global_row = global_row_base + m * TILE_M;
      const int tiled_col = base + vec_col;

      float4 vec = (float4) (0, 0, 0, 0);

      if (global_row < M && vec_col < TILE_K) {
        if (tiled_col + 3 < K) {
          vec = vload4(0, &A[global_row * K + tiled_col]);
        } else {
          float tmp[4] = {0};
#pragma unroll
          for (int i = 0; i < 4; i++) {
            int col = tiled_col + i;
            if (col < K) {
              tmp[i] = A[global_row * K + col];
            }
          }
          vec = (float4) (tmp[0], tmp[1], tmp[2], tmp[3]);
        }

        // transpose 写入
        shmem_a[m][(vec_col + 0) * (TILE_M + 1) + local_row] = vec.x;
        shmem_a[m][(vec_col + 1) * (TILE_M + 1) + local_row] = vec.y;
        shmem_a[m][(vec_col + 2) * (TILE_M + 1) + local_row] = vec.z;
        shmem_a[m][(vec_col + 3) * (TILE_M + 1) + local_row] = vec.w;
      } else {
        // zero padding
        shmem_a[m][(vec_col + 0) * (TILE_M + 1) + local_row] = 0.0f;
        shmem_a[m][(vec_col + 1) * (TILE_M + 1) + local_row] = 0.0f;
        shmem_a[m][(vec_col + 2) * (TILE_M + 1) + local_row] = 0.0f;
        shmem_a[m][(vec_col + 3) * (TILE_M + 1) + local_row] = 0.0f;
      }
    }

#pragma unroll
    // load B(k, j ~ j + REG_TILE_N) in shared memory
    for (int n = 0; n < REG_TILE_N; ++n) {
      const int tiled_row = base + local_row;
      const int global_col = global_col_base + n * TILE_N;
      if (tiled_row < K && global_col < N) {
        shmem_b[n][local_row * (TILE_N + 1) + local_col] =
            B[tiled_row * N + global_col];
      } else {
        shmem_b[n][local_row * (TILE_N + 1) + local_col] = 0.0f;
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // compute
    for (int k = 0; k < TILE_K; ++k) {
#pragma unroll
      for (int m = 0; m < REG_TILE_M; ++m) {
        frag_a[m] = shmem_a[m][k * (TILE_M + 1) + local_row];
      }

#pragma unroll
      for (int n = 0; n < REG_TILE_N; ++n) {
        frag_b[n] = shmem_b[n][k * (TILE_N + 1) + local_col];
      }
#pragma unroll
      for (int m = 0; m < REG_TILE_M; ++m) {
        float a_val = frag_a[m];
#pragma unroll
        for (int n = 0; n < REG_TILE_N; ++n) {
          sum[m * REG_TILE_N + n] += a_val * frag_b[n];
        }
      }
    }

    // sync
    barrier(CLK_LOCAL_MEM_FENCE);
  }

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
          C[idx] = alpha * sum[m * REG_TILE_N + n] + beta * C[idx];
        }
      }
    }
  }
}
