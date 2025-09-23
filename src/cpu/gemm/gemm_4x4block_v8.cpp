#include "gemm.h"

static void addDot4x4(int k, float *A, int lda, float *B, int ldb, float *C,
                      int ldc) {
  float c0_reg_0 = 0.0f, c0_reg_1 = 0.0f, c0_reg_2 = 0.0f, c0_reg_3 = 0.0f;
  float c1_reg_0 = 0.0f, c1_reg_1 = 0.0f, c1_reg_2 = 0.0f, c1_reg_3 = 0.0f;
  float c2_reg_0 = 0.0f, c2_reg_1 = 0.0f, c2_reg_2 = 0.0f, c2_reg_3 = 0.0f;
  float c3_reg_0 = 0.0f, c3_reg_1 = 0.0f, c3_reg_2 = 0.0f, c3_reg_3 = 0.0f;

  float a0_reg = 0.0f, a1_reg = 0.0f, a2_reg = 0.0f, a3_reg = 0.0f;

  float *bp0_ptr = &B(0, 0);
  float *bp1_ptr = &B(0, 1);
  float *bp2_ptr = &B(0, 2);
  float *bp3_ptr = &B(0, 3);

  float b0_reg = 0.0f, b1_reg = 0.0f, b2_reg = 0.0f, b3_reg = 0.0f;

  for (int p = 0; p < k; ++p) {
    a0_reg = A(0, p);

    b0_reg = *bp0_ptr;
    b1_reg = *bp1_ptr;
    b2_reg = *bp2_ptr;
    b3_reg = *bp3_ptr;
    
    bp0_ptr += ldb;
    bp1_ptr += ldb;
    bp2_ptr += ldb;
    bp3_ptr += ldb;
    
    c0_reg_0 += a0_reg * b0_reg;
    c0_reg_1 += a0_reg * b1_reg;
    c0_reg_2 += a0_reg * b2_reg;
    c0_reg_3 += a0_reg * b3_reg;

    a1_reg = A(1, p);
    c1_reg_0 += a1_reg * b0_reg;
    c1_reg_1 += a1_reg * b1_reg;
    c1_reg_2 += a1_reg * b2_reg;
    c1_reg_3 += a1_reg * b3_reg;

    a2_reg = A(2, p);
    c2_reg_0 += a2_reg * b0_reg;
    c2_reg_1 += a2_reg * b1_reg;
    c2_reg_2 += a2_reg * b2_reg;
    c2_reg_3 += a2_reg * b3_reg;

    a3_reg = A(3, p);
    c3_reg_0 += a3_reg * b0_reg;
    c3_reg_1 += a3_reg * b1_reg;
    c3_reg_2 += a3_reg * b2_reg;
    c3_reg_3 += a3_reg * b3_reg;
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

void gemm_4x4block_v8(float *A, float *B, float *C, int m, int n, int k) {
  int lda = k, ldb = n, ldc = n;
  for (int i = 0; i < m; i += 4) {
    for (int j = 0; j < n; j += 4) {
      addDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}