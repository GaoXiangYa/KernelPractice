#define TILE_K 4

__kernel void gemm_v1_kernel(__global const float* A, __global const float* B,
                             __global float* C, const int M, const int N,
                             const int K, float alpha, float beta) {
  const int row = get_global_id(0);
  const int col = get_global_id(1);
  if (row >= M || col >= N)
    return;
  float sum = 0.0f;
  for (int p = 0; p < K; p += TILE_K) {
    float4 vec_a = vload4(0, &A[row * K + p]);
    float4 vec_b = {B[p * N + col], B[(p + 1) * N + col], B[(p + 2) * N + col], B[(p + 3) * N + col]};
    sum += (vec_a.x * vec_b.x + vec_a.y * vec_b.y + vec_a.z * vec_b.z +
            vec_a.w * vec_b.w);
  }
  C[row * N + col] = alpha * sum + beta * C[row * N + col];
}