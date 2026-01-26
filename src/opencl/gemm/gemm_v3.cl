#define TILE_K 32
#define COARSE_FACTOR 8
#define TILE_VEC 4

static inline float mul_vec(__local float* vec_a, __local float* vec_b,
                            const int local_row, const int local_col) {
  float sum = 0;
  // #pragma unroll
  for (int t = 0; t < TILE_K; t += TILE_VEC) {
    float4 val_a = vload4(0, &vec_a[local_row * TILE_K + t]);
    float4 val_b = {vec_b[t * TILE_K + local_col],
                    vec_b[(t + 1) * TILE_K + local_col],
                    vec_b[(t + 2) * TILE_K + local_col],
                    vec_b[(t + 3) * TILE_K + local_col]};
    sum += (val_a.x * val_b.x + val_a.y * val_b.y + val_a.z * val_b.z +
            val_a.w * val_b.w);
  }
  return sum;
}

__kernel void gemm_v3_kernel(__global const float* A, __global const float* B,
                             __global float* C, const int M, const int N,
                             const int K, float alpha, float beta) {
  __local float shared_A[TILE_K * TILE_K];
  __local float shared_B[TILE_K * TILE_K * COARSE_FACTOR];

  int local_col = get_local_id(0);
  int local_row = get_local_id(1);

  int global_col = TILE_K * get_group_id(0) + local_col;
  int global_row = TILE_K * get_group_id(1) + local_row;
  float sum[COARSE_FACTOR];

#pragma unroll
  for (int i = 0; i < COARSE_FACTOR; ++i) {
    sum[i] = 0.0f;
  }

  float num_tile = (K + TILE_K - 1) / TILE_K;
  for (int ph = 0; ph < num_tile; ++ph) {
    int base = ph * TILE_K;
    int a_col = base + local_col;

    if (global_row < M && a_col < K) {
      shared_A[local_row * TILE_K + local_col] = A[global_row * K + a_col];
    } else {
      shared_A[local_row * TILE_K + local_col] = 0.0f;
    }

    int b_row = base + local_row;
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      if (b_row < K && global_col < N) {
        shared_B[c * TILE_K * TILE_K + local_row * TILE_K + local_col] =
            B[b_row * N + global_col];
      } else {
        shared_B[c * TILE_K * TILE_K + local_row * TILE_K + local_col] = 0.0f;
      }
    }

    // sync
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int c = 0; c < COARSE_FACTOR; ++c) {
      sum[c] += mul_vec(shared_A, shared_B + c * TILE_K * TILE_K, local_row,
                        local_col);
    }
    // sync
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (global_row < M && global_col < N) {
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      C[global_row * N + global_col] =
          alpha * sum[c] + beta * C[global_row * N + global_col];
    }
  }
}