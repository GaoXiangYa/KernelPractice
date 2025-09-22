#include "gemm.h"

static inline void addDot1x4(int k, float *A, int lda, float *B, int ldb, float *C, int ldc) {
  float c_reg_0 = 0.0, c_reg_1 = 0.0, c_reg_2 = 0.0, c_reg_3 = 0.0f;
  float a_reg = 0;
  for (int p = 0; p < k; ++ p) {
    a_reg = A(0, p);
    c_reg_0 += a_reg * B(p, 0);
    c_reg_1 += a_reg * B(p, 1);
    c_reg_2 += a_reg * B(p, 2);
    c_reg_3 += a_reg * B(p, 3);
  }
  C(0, 0) += c_reg_0;
  C(0, 1) += c_reg_1;
  C(0, 2) += c_reg_2;
  C(0, 3) += c_reg_3;
}

void gemm_v5(float *A, float *B, float *C, int m, int n, int k) {
  int lda = k, ldb = n, ldc = n;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; j += 4) {
      addDot1x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}