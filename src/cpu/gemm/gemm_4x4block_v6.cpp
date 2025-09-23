#include "gemm.h"

static void addDot4x4(int k, float *A, int lda, float *B, int ldb, float *C,
                      int ldc) {
  float c0_reg_0 = 0.0f, c0_reg_1 = 0.0f, c0_reg_2 = 0.0f, c0_reg_3 = 0.0f;
  float c1_reg_0 = 0.0f, c1_reg_1 = 0.0f, c1_reg_2 = 0.0f, c1_reg_3 = 0.0f;
  float c2_reg_0 = 0.0f, c2_reg_1 = 0.0f, c2_reg_2 = 0.0f, c2_reg_3 = 0.0f;
  float c3_reg_0 = 0.0f, c3_reg_1 = 0.0f, c3_reg_2 = 0.0f, c3_reg_3 = 0.0f;
  float a0_reg = 0.0f, a1_reg = 0.0f, a2_reg = 0.0f, a3_reg = 0.0f;

  for (int p = 0; p < k; ++p) {
    a0_reg = A(0, p);
    c0_reg_0 += a0_reg * B(p, 0);
    c0_reg_1 += a0_reg * B(p, 1);
    c0_reg_2 += a0_reg * B(p, 2);
    c0_reg_3 += a0_reg * B(p, 3);

    a1_reg = A(1, p);
    c1_reg_0 += a1_reg * B(p, 0);
    c1_reg_1 += a1_reg * B(p, 1);
    c1_reg_2 += a1_reg * B(p, 2);
    c1_reg_3 += a1_reg * B(p, 3);

    a2_reg = A(2, p);
    c2_reg_0 += a2_reg * B(p, 0);
    c2_reg_1 += a2_reg * B(p, 1);
    c2_reg_2 += a2_reg * B(p, 2);
    c2_reg_3 += a2_reg * B(p, 3);

    a3_reg = A(3, p);
    c3_reg_0 += a3_reg * B(p, 0);
    c3_reg_1 += a3_reg * B(p, 1);
    c3_reg_2 += a3_reg * B(p, 2);
    c3_reg_3 += a3_reg * B(p, 3);
  }

  C(0, 0) += c0_reg_0;
  C(0, 1) += c0_reg_1;
  C(0, 2) += c0_reg_2;
  C(0, 3) += c0_reg_3;

  C(1, 0) += c1_reg_0;
  C(1, 1) += c1_reg_1;
  C(1, 2) += c1_reg_2;
  C(1, 3) += c1_reg_3;

  C(2, 0) += c2_reg_0;
  C(2, 1) += c2_reg_1;
  C(2, 2) += c2_reg_2;
  C(2, 3) += c2_reg_3;

  C(3, 0) += c3_reg_0;
  C(3, 1) += c3_reg_1;
  C(3, 2) += c3_reg_2;
  C(3, 3) += c3_reg_3;
}

void gemm_4x4block_v6(float *A, float *B, float *C, int m, int n, int k) {
  int lda = k, ldb = n, ldc = n;
  for (int i = 0; i < m; i += 4) {
    for (int j = 0; j < n; j += 4) {
      addDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}