#define VEC 4
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// A[M x K], B[K x N], C[M x N]
__kernel void gemm_v4_kernel(__global const float4* A, __global const float4* B,
                             __global float4* C, const int M, const int N,
                             const int K, float alpha, float beta) {
  const int local_row = get_local_id(1);
  const int local_col = get_local_id(0);

  const int global_row = TILE_M * get_group_id(1) + local_row;
  const int global_col = TILE_N / VEC * get_group_id(0) + local_col;

  __local float4 shmem_a[TILE_M * TILE_K / VEC];
  __local float4 shmem_b[TILE_K * TILE_N / VEC];

  float4 sum = (float4) (0.0f);

  const int num_tiles = (K + TILE_K - 1) / TILE_K;

  for (int ph = 0; ph < num_tiles; ++ph) {
    const int tile_k_vec = ph * (TILE_K / VEC);
    if (global_row < M && local_col < TILE_K / VEC) {
      shmem_a[local_row * TILE_K / VEC + local_col] =
          A[global_row * (K / VEC) + tile_k_vec + local_col];
    } else {
      shmem_a[local_row * TILE_K / VEC + local_col] = (float4) (0.0f);
    }

    int b_row = ph * TILE_K + local_row;
    if (b_row < K && global_col < N / VEC) {
      shmem_b[local_row * TILE_N / VEC + local_col] =
          B[b_row * (N / VEC) + global_col];
    } else {
      shmem_b[local_row * TILE_N / VEC + local_col] = (float4) (0.0f);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float4 a_vec, b_vec;
    float a_val;
    for (int k = 0; k < TILE_K; k += 4) {
      float4 a_vec = shmem_a[local_row * (TILE_K / VEC) + k / VEC];

      float4 b0 = shmem_b[(k + 0) * (TILE_N / VEC) + local_col];
      float4 b1 = shmem_b[(k + 1) * (TILE_N / VEC) + local_col];
      float4 b2 = shmem_b[(k + 2) * (TILE_N / VEC) + local_col];
      float4 b3 = shmem_b[(k + 3) * (TILE_N / VEC) + local_col];

      sum += a_vec.x * b0;
      sum += a_vec.y * b1;
      sum += a_vec.z * b2;
      sum += a_vec.w * b3;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (global_row < M && global_col < N / VEC) {
    int idx = global_row * (N / VEC) + global_col;
    float4 c_old = C[idx];
    C[idx] = alpha * sum + beta * c_old;
  }
}