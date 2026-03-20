#define VEC_WIDTH 4
#define TILE_M 32
#define TILE_N 32
#define TILE_K 32
#define COARSE_FACTOR 4

// A[M x K], B[K x N], C[M x N]
// shared memory + thread coarse + vectorization + transpose shmem_b
__kernel void gemm_v6_kernel(__global const float4* A, __global const float4* B,
                             __global float4* C, const int M, const int N,
                             const int K, float alpha, float beta) {
  __local float4 shmem_a[TILE_M * (TILE_K / VEC_WIDTH + 1)];
  __local float4 shmem_b[COARSE_FACTOR][(TILE_N / VEC_WIDTH) * (TILE_K + 1)];

  const int local_row = get_local_id(1);
  const int local_col = get_local_id(0);

  const int global_row = TILE_M * get_group_id(1) + local_row;
  const int global_col_base =
      TILE_N / VEC_WIDTH * get_group_id(0) * COARSE_FACTOR + local_col;

  float4 sum[COARSE_FACTOR];
#pragma unroll
  for (int c = 0; c < COARSE_FACTOR; ++c) {
    sum[c] = (float4) (0.0f);
  }

  const int num_tiles = (K + TILE_K - 1) / TILE_K;
  for (int ph = 0; ph < num_tiles; ++ph) {
    const int tiled_col = ph * (TILE_K / VEC_WIDTH) + local_col;
    // load matrix A
    if (global_row < M && local_col < (K / VEC_WIDTH)) {
      shmem_a[local_row * (TILE_K / VEC_WIDTH + 1) + local_col] =
          A[global_row * (K / VEC_WIDTH) + tiled_col];
    } else {
      shmem_a[local_row * (TILE_K / VEC_WIDTH + 1) + local_col] = (float4) (0.0f);
    }

    // load matrix B
    const int tiled_row = ph * TILE_K + local_row;
#pragma unroll
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      const int global_col = global_col_base + c * TILE_N / VEC_WIDTH;
      if (tiled_row < K && global_col < (N / VEC_WIDTH)) {
        shmem_b[c][local_col * (TILE_K + 1)+ local_row] =
            B[tiled_row * (N / VEC_WIDTH) + global_col];
      } else {
        shmem_b[c][local_col * (TILE_K + 1) + local_row] = (float4) (0.0f);
      }
    }

    // sync
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 a_vec;
    float4 b0, b1, b2, b3;
#pragma unroll
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      for (int k = 0; k < TILE_K; k += VEC_WIDTH) {
        a_vec = shmem_a[local_row * (TILE_K / VEC_WIDTH + 1) + k / VEC_WIDTH];
        b0 = shmem_b[c][local_col * (TILE_K + 1) + (k + 0)];
        b1 = shmem_b[c][local_col * (TILE_K + 1) + (k + 1)];
        b2 = shmem_b[c][local_col * (TILE_K + 1) + (k + 2)];
        b3 = shmem_b[c][local_col * (TILE_K + 1) + (k + 3)];

        sum[c] += a_vec.x * b0;
        sum[c] += a_vec.y * b1;
        sum[c] += a_vec.z * b2;
        sum[c] += a_vec.w * b3;
      }
    }
    // sync
    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (int c = 0; c < COARSE_FACTOR; ++c) {
    int global_col = global_col_base + c * TILE_N / VEC_WIDTH;
    if (global_row < M && global_col < N / VEC_WIDTH) {
      int idx = global_row * (N / VEC_WIDTH) + global_col;
      float4 c_old = C[idx];
      C[idx] = alpha * sum[c] + beta * c_old;
    }
  }
}