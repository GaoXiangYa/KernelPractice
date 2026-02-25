#define TILE_K 16
#define COARSE_FACTOR 4
#define TILE_VEC 4
#define DATA_WIDTH 4

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

__kernel void gemm_v4_kernel(__global const float4* A, __global const float4* B,
                             __global float4* C, const int M, const int N,
                             const int K, float alpha, float beta) {
  __local float4 shared_A[TILE_K * TILE_K / DATA_WIDTH];
  __local float4 shared_B[COARSE_FACTOR][TILE_K * TILE_K / DATA_WIDTH];

  int local_col = get_local_id(0);
  int local_row = get_local_id(1);

  int global_col_base = TILE_K * get_group_id(0) * COARSE_FACTOR + local_col;
  int global_row = TILE_K * get_group_id(1) + local_row;
  float4 sum[COARSE_FACTOR];

  // #pragma unroll
  for (int i = 0; i < COARSE_FACTOR; ++i) {
    sum[i].x = 0.0f;
    sum[i].y = 0.0f;
    sum[i].z = 0.0f;
    sum[i].w = 0.0f;
  }

  float num_tile = (K + TILE_K - 1) / TILE_K;
  for (int ph = 0; ph < num_tile; ++ph) {
    int base = ph * TILE_K;
    int a_col = base + local_col;

    if (global_row < M && a_col < K) {
      shared_A[local_row * TILE_K + local_col] =
          A[(global_row / DATA_WIDTH) * K + a_col];
    } else {
      shared_A[local_row * TILE_K + local_col] = 0.0f;
    }

    int b_row = base + local_row;
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      int global_col = global_col_base + c * TILE_K;
      int local_b_base = c * TILE_K * TILE_K;
      if (b_row < K && global_col < N) {
        shared_B[c][local_row * TILE_K + local_col] =
            B[(b_row / DATA_WIDTH) * N + global_col];
      } else {
        shared_B[c][local_row * TILE_K + local_col] = 0.0f;
      }
    }

    // sync
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TILE_K; ++k) {
      float4 a_vec = shared_A[local_row * TILE_K + k];
#pragma unroll
      for (int c = 0; c < COARSE_FACTOR; ++c) {
        float4 b_vec = shared_B[c][k * TILE_K + local_col];
        sum[c].x += a_vec.x * b_vec.x;
        sum[c].y += a_vec.y * b_vec.y;
        sum[c].z += a_vec.z * b_vec.z;
        sum[c].w += a_vec.w * b_vec.w;
      }
    }
    // sync
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (global_row < M) {
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      int global_col = global_col_base + c * TILE_K;
      if (global_col < N) {
        C[global_row * N + global_col].x =
            alpha * sum[c].x + beta * C[global_row * N + global_col].x;
        
        C[global_row * N + global_col].y =
            alpha * sum[c].y + beta * C[global_row * N + global_col].y;

        C[global_row * N + global_col].z =
            alpha * sum[c].z + beta * C[global_row * N + global_col].z;

        C[global_row * N + global_col].w =
            alpha * sum[c].w + beta * C[global_row * N + global_col].w;
      }
    }
  }
}