__kernel void gemm_v0_kernel(__global const float* A, __global const float* B,
                        __global float* C, const int M, const int N,
                        const int K, float alpha, float beta) {
  const int row = get_global_id(0);
  const int col = get_global_id(1);

  if (row >= M || col >= N)
    return;

  float sum = 0.0f;
  for (int p = 0; p < K; ++p) {
    sum += A[row * K + p] * B[p * N + col];
  }

  C[row * N + col] = alpha * sum + beta * C[row * N + col];
}