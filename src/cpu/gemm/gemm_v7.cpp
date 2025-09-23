#include "gemm.h"

static inline void addDot1x4(int k, float *A, int lda, float *B, int ldb, float *C, int ldc) {
  float c_reg_0 = 0.0, c_reg_1 = 0.0, c_reg_2 = 0.0, c_reg_3 = 0.0f;
  float a_reg = 0;
  float* bp0_ptr = &B(0, 0);
  float* bp1_ptr = &B(0, 1);
  float* bp2_ptr = &B(0, 2);
  float* bp3_ptr = &B(0, 3);

  for (int p = 0; p < k; p += 4) {
    a_reg = A(0, p);
    c_reg_0 += a_reg * *bp0_ptr;
    bp0_ptr += ldb;
    c_reg_1 += a_reg * *bp1_ptr;
    bp1_ptr += ldb;
    c_reg_2 += a_reg * *bp2_ptr;
    bp2_ptr += ldb;
    c_reg_3 += a_reg * *bp3_ptr;
    bp3_ptr += ldb;
    
    a_reg = A(0, p + 1);
    c_reg_0 += a_reg * *bp0_ptr;
    bp0_ptr += ldb;
    c_reg_1 += a_reg * *bp1_ptr;
    bp1_ptr += ldb;
    c_reg_2 += a_reg * *bp2_ptr;
    bp2_ptr += ldb;
    c_reg_3 += a_reg * *bp3_ptr;
    bp3_ptr += ldb;

    a_reg = A(0, p + 2);
    c_reg_0 += a_reg * *bp0_ptr;
    bp0_ptr += ldb;
    c_reg_1 += a_reg * *bp1_ptr;
    bp1_ptr += ldb;
    c_reg_2 += a_reg * *bp2_ptr;
    bp2_ptr += ldb;
    c_reg_3 += a_reg * *bp3_ptr;
    bp3_ptr += ldb;


    a_reg = A(0, p + 3);
    c_reg_0 += a_reg * *bp0_ptr;
    bp0_ptr += ldb;
    c_reg_1 += a_reg * *bp1_ptr;
    bp1_ptr += ldb;
    c_reg_2 += a_reg * *bp2_ptr;
    bp2_ptr += ldb;
    c_reg_3 += a_reg * *bp3_ptr;
    bp3_ptr += ldb;
  }

  C(0, 0) += c_reg_0;
  C(0, 1) += c_reg_1;
  C(0, 2) += c_reg_2;
  C(0, 3) += c_reg_3;
}

void gemm_v7(float *A, float *B, float *C, int m, int n, int k) {
  int lda = k, ldb = n, ldc = n;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; j += 4) {
      addDot1x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}