#define MICRO_SIZE 8
#define BK 16
#define BM (BK * MICRO_SIZE)
#define BN (BK * MICRO_SIZE)
#define BK_PAD (BK + 1)

// C[M, N] = A[M, K] * B[K, N], 8x8x8 tiling
// Each work-item computes an 8x8 micro-tile and uses shared memory to cache
// one K-tile of A and one K-tile of B.
__kernel void gemm_v5_kernel(__global const float* A, __global const float* B,
                             __global float* C, const int M, const int N,
                             const int K, float alpha, float beta) {
  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  const int local_wg_x = get_local_size(0);
  const int local_wg_y = get_local_size(1);
  const int group_x = get_group_id(0);
  const int group_y = get_group_id(1);
  const int local_x = get_local_id(0);
  const int local_y = get_local_id(1);

  const int tile_col_base = group_x * local_wg_x * MICRO_SIZE;
  const int tile_row_base = group_y * local_wg_y * MICRO_SIZE;
  const int thread_col_base = local_x * MICRO_SIZE;
  const int thread_row_base = local_y * MICRO_SIZE;

  __local float sh_a[BM * BK_PAD];
  __local float sh_b[BK_PAD * BN];

  float accum[MICRO_SIZE * MICRO_SIZE] = {0.0f};

  for (int k_base = 0; k_base < K; k_base += BK) {
    #pragma unroll
    for (int row_offset = 0; row_offset < MICRO_SIZE; ++row_offset) {
      const int shared_row = thread_row_base + row_offset;
      const int global_row = tile_row_base + shared_row;
      const int global_col = k_base + local_x;
      if (global_row < M && global_col < K) {
        sh_a[shared_row * BK_PAD + local_x] = A[global_row * lda + global_col];
      } else {
        sh_a[shared_row * BK_PAD + local_x] = 0.0f;
      }
    }

    #pragma unroll
    for (int col_offset = 0; col_offset < MICRO_SIZE; ++col_offset) {
      const int shared_col = thread_col_base + col_offset;
      const int global_col = tile_col_base + shared_col;
      const int global_row = k_base + local_y;
      if (global_row < K && global_col < N) {
        sh_b[local_y * (BN + 1) + shared_col] = B[global_row * ldb + global_col];
      } else {
        sh_b[local_y * (BN + 1) + shared_col] = 0.0f;
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int ik = 0; ik < BK; ++ik) {
      #pragma unroll
      for (int row_offset = 0; row_offset < MICRO_SIZE; ++row_offset) {
        const float a_value = sh_a[(thread_row_base + row_offset) * BK_PAD + ik];
        #pragma unroll
        for (int col_offset = 0; col_offset < MICRO_SIZE; ++col_offset) {
          const int shared_col = thread_col_base + col_offset;
          accum[row_offset * MICRO_SIZE + col_offset] +=
              a_value * sh_b[ik * (BN + 1) + shared_col];
        }
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  #pragma unroll
  for (int row_offset = 0; row_offset < MICRO_SIZE; ++row_offset) {
    const int global_row = tile_row_base + thread_row_base + row_offset;
    #pragma unroll
    for (int col_offset = 0; col_offset < MICRO_SIZE; ++col_offset) {
      const int global_col = tile_col_base + thread_col_base + col_offset;
      if (global_row < M && global_col < N) {
        C[global_row * ldc + global_col] =
            alpha * accum[row_offset * MICRO_SIZE + col_offset] +
            beta * C[global_row * ldc + global_col];
      }
    }
  }
}