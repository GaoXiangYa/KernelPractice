#include <gtest/gtest.h>
#include <vector>
#include "gemm.h"
#include "utils.h"

void gemm_ref(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = alpha * sum + beta * C[m * N + n];
    }
  }
}

TEST(GEMM, gemm_v0) {
  constexpr int M = 2048;
  constexpr int N = 128;
  constexpr int K = 512;

  std::vector<float> A(M * K, 0.0f);
  std::vector<float> B(K * N, 0.0f);
  std::vector<float> C_cpu(M * N, 0.0f);
  std::vector<float> C_ocl(M * N, 0.0f);

  set_random_values(A, -1.0f, 1.0f);
  set_random_values(B, -1.0f, 1.0f);

  gemm_ref(A.data(), B.data(), C_cpu.data(), M, N, K);
  gemm_v0(A.data(), B.data(), C_ocl.data(), M, N, K);

  constexpr float kEpsilon = 1e-3f;
  for (int i = 0; i < M * N; ++i) {
    EXPECT_NEAR(C_ocl[i], C_cpu[i], kEpsilon);
  }
}