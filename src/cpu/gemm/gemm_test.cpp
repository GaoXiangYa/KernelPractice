#include "gemm.h"
#include "util.h"
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>

TEST(gemm, gemm_v0) {
  const int m = 256, n = 256, k = 256;
  const float random_min = -1.0f, random_max = 1.0f;

  float *A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));
  float *BLAS_A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));

  initMatrix(A, m, k, random_min, random_max);
  std::memcpy(BLAS_A, A, m * k * sizeof(float));

  float *B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));
  float *BLAS_B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));

  initMatrix(B, k, n, random_min, random_max);
  std::memcpy(BLAS_B, B, k * n * sizeof(float));

  float *C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));
  float *BLAS_C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));

  initMatrix(C, m, n, random_min, random_max);
  std::memcpy(BLAS_C, C, m * n * sizeof(float));

  gemm_v0(A, B, C, m, n, k);
  gemm_blas(BLAS_A, BLAS_B, BLAS_C, m, n, k);

  const float tolerance = 0.001f;
  for (int i = 0; i < m * n; ++ i) {
    EXPECT_LE(std::abs(C[i] - BLAS_C[i]), tolerance);
  }
}

TEST(gemm, gemm_v1) {
  const int m = 256, n = 256, k = 256;
  const float random_min = -1.0f, random_max = 1.0f;

  float *A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));
  float *BLAS_A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));

  initMatrix(A, m, k, random_min, random_max);
  std::memcpy(BLAS_A, A, m * k * sizeof(float));

  float *B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));
  float *BLAS_B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));

  initMatrix(B, k, n, random_min, random_max);
  std::memcpy(BLAS_B, B, k * n * sizeof(float));

  float *C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));
  float *BLAS_C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));

  initMatrix(C, m, n, random_min, random_max);
  std::memcpy(BLAS_C, C, m * n * sizeof(float));

  gemm_v1(A, B, C, m, n, k);
  gemm_blas(BLAS_A, BLAS_B, BLAS_C, m, n, k);

  const float tolerance = 0.001f;
  for (int i = 0; i < m * n; ++ i) {
    EXPECT_LE(std::abs(C[i] - BLAS_C[i]), tolerance);
  }
}

TEST(gemm, gemm_v2) {
  const int m = 256, n = 256, k = 256;
  const float random_min = -1.0f, random_max = 1.0f;

  float *A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));
  float *BLAS_A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));

  initMatrix(A, m, k, random_min, random_max);
  std::memcpy(BLAS_A, A, m * k * sizeof(float));

  float *B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));
  float *BLAS_B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));

  initMatrix(B, k, n, random_min, random_max);
  std::memcpy(BLAS_B, B, k * n * sizeof(float));

  float *C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));
  float *BLAS_C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));

  initMatrix(C, m, n, random_min, random_max);
  std::memcpy(BLAS_C, C, m * n * sizeof(float));

  gemm_v2(A, B, C, m, n, k);
  gemm_blas(BLAS_A, BLAS_B, BLAS_C, m, n, k);

  const float tolerance = 0.001f;
  for (int i = 0; i < m * n; ++ i) {
    EXPECT_LE(std::abs(C[i] - BLAS_C[i]), tolerance);
  }
}

TEST(gemm, gemm_v3) {
  const int m = 256, n = 256, k = 256;
  const float random_min = -1.0f, random_max = 1.0f;

  float *A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));
  float *BLAS_A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));

  initMatrix(A, m, k, random_min, random_max);
  std::memcpy(BLAS_A, A, m * k * sizeof(float));

  float *B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));
  float *BLAS_B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));

  initMatrix(B, k, n, random_min, random_max);
  std::memcpy(BLAS_B, B, k * n * sizeof(float));

  float *C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));
  float *BLAS_C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));

  initMatrix(C, m, n, random_min, random_max);
  std::memcpy(BLAS_C, C, m * n * sizeof(float));

  gemm_v3(A, B, C, m, n, k);
  gemm_blas(BLAS_A, BLAS_B, BLAS_C, m, n, k);

  const float tolerance = 0.001f;
  for (int i = 0; i < m * n; ++ i) {
    EXPECT_LE(std::abs(C[i] - BLAS_C[i]), tolerance);
  }
}

TEST(gemm, gemm_v4) {
  const int m = 256, n = 256, k = 256;
  const float random_min = -1.0f, random_max = 1.0f;

  float *A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));
  float *BLAS_A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));

  initMatrix(A, m, k, random_min, random_max);
  std::memcpy(BLAS_A, A, m * k * sizeof(float));

  float *B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));
  float *BLAS_B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));

  initMatrix(B, k, n, random_min, random_max);
  std::memcpy(BLAS_B, B, k * n * sizeof(float));

  float *C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));
  float *BLAS_C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));

  initMatrix(C, m, n, random_min, random_max);
  std::memcpy(BLAS_C, C, m * n * sizeof(float));

  gemm_v4(A, B, C, m, n, k);
  gemm_blas(BLAS_A, BLAS_B, BLAS_C, m, n, k);

  const float tolerance = 0.001f;
  for (int i = 0; i < m * n; ++ i) {
    EXPECT_LE(std::abs(C[i] - BLAS_C[i]), tolerance);
  }
}

TEST(gemm, gemm_v5) {
  const int m = 256, n = 256, k = 256;
  const float random_min = -1.0f, random_max = 1.0f;

  float *A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));
  float *BLAS_A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));

  initMatrix(A, m, k, random_min, random_max);
  std::memcpy(BLAS_A, A, m * k * sizeof(float));

  float *B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));
  float *BLAS_B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));

  initMatrix(B, k, n, random_min, random_max);
  std::memcpy(BLAS_B, B, k * n * sizeof(float));

  float *C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));
  float *BLAS_C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));

  initMatrix(C, m, n, random_min, random_max);
  std::memcpy(BLAS_C, C, m * n * sizeof(float));

  gemm_v5(A, B, C, m, n, k);
  gemm_blas(BLAS_A, BLAS_B, BLAS_C, m, n, k);

  const float tolerance = 0.001f;
  for (int i = 0; i < m * n; ++ i) {
    EXPECT_LE(std::abs(C[i] - BLAS_C[i]), tolerance);
  }
}

TEST(gemm, gemm_v6) {
  const int m = 256, n = 256, k = 256;
  const float random_min = -1.0f, random_max = 1.0f;

  float *A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));
  float *BLAS_A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));

  initMatrix(A, m, k, random_min, random_max);
  std::memcpy(BLAS_A, A, m * k * sizeof(float));

  float *B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));
  float *BLAS_B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));

  initMatrix(B, k, n, random_min, random_max);
  std::memcpy(BLAS_B, B, k * n * sizeof(float));

  float *C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));
  float *BLAS_C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));

  initMatrix(C, m, n, random_min, random_max);
  std::memcpy(BLAS_C, C, m * n * sizeof(float));

  gemm_v6(A, B, C, m, n, k);
  gemm_blas(BLAS_A, BLAS_B, BLAS_C, m, n, k);

  const float tolerance = 0.001f;
  for (int i = 0; i < m * n; ++ i) {
    EXPECT_LE(std::abs(C[i] - BLAS_C[i]), tolerance);
  }
}