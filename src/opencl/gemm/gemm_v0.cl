#define GEMM_A(i, j) A[(i) * lda + (j)]
#define GEMM_B(i, j) B[(i) * ldb + (j)]
#define GEMM_C(i, j) C[(i) * ldc + (j)]

// C[M, N] = A[M, K] * B[K, N]
__kernel void gemm_v0_kernel(__global const float* A, __global const float* B,
                        __global float* C, const int M, const int N,
                        const int K, float alpha, float beta) {
  const int lda =  K, ldb = N, ldc = N;
  const int gx = get_global_id(0);
  const int gy = get_global_id(1);
  float sum = 0.0f;
  for (int k = 0; k < K; ++ k) {
    sum += GEMM_A(gy, k) * GEMM_B(k, gx);
  }
  GEMM_C(gy, gx) = alpha * sum + beta * GEMM_C(gy, gx);
}