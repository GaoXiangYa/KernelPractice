#define TILE_K 16
#define COARSE_FACTOR 4
#define TILE_VEC 4 

// thread coarse + thread_memory
__kernel void gemm_v3_kernel(__global const float* A, __global const float* B,
                             __global float* C, const int M, const int N,
                             const int K, float alpha, float beta) {
  __local float shared_A[TILE_K * (TILE_K + 1)];
  __local float shared_B[COARSE_FACTOR][TILE_K * (TILE_K + 1)];

  int local_col = get_local_id(0);
  int local_row = get_local_id(1);

  int global_col_base = TILE_K * get_group_id(0) * COARSE_FACTOR + local_col;
  int global_row = TILE_K * get_group_id(1) + local_row;
  float sum[COARSE_FACTOR];

// #pragma unroll
  for (int i = 0; i < COARSE_FACTOR; ++i) {
    sum[i] = 0.0f;
  }

  float num_tile = (K + TILE_K - 1) / TILE_K;
  for (int ph = 0; ph < num_tile; ++ph) {
    int base = ph * TILE_K;
    int a_col = base + local_col;

    if (global_row < M && a_col < K) {
      shared_A[local_row * (TILE_K + 1) + local_col] = A[global_row * K + a_col];
    } else {
      shared_A[local_row * (TILE_K + 1) + local_col] = 0.0f;
    }

    int b_row = base + local_row;
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      int global_col = global_col_base + c * TILE_K;
      int local_b_base = c * TILE_K * TILE_K;
      if (b_row < K && global_col < N) {
        shared_B[c][local_row * (TILE_K + 1) + local_col] =
            B[b_row * N + global_col];
      } else {
        shared_B[c][local_row * (TILE_K + 1) + local_col] = 0.0f;
      }
    }

    // sync
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TILE_K; ++ k) {
      float a_val = shared_A[local_row * (TILE_K + 1)+ k];
    #pragma unroll
      for (int c = 0; c < COARSE_FACTOR; ++ c) {
        float b_val = shared_B[c][k * (TILE_K + 1) + local_col];
        sum[c] += a_val * b_val;
      }
    }
    // sync
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (global_row < M) {
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      int global_col = global_col_base + c * TILE_K;
      if (global_col < N) {
        C[global_row * N + global_col] =
            alpha * sum[c] + beta * C[global_row * N + global_col];
      }
    }
  }
}