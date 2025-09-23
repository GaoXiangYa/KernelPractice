#include "gemm.h"

static void addDot4x4(int k, float *A, int lda, float *B, int ldb, float *C, int ldc) {
  addDot(k, &A[0], &B[0], ldb, &C[0]);
  addDot(k, &A[0], &B[1], ldb, &C[1]);
  addDot(k, &A[0], &B[2], ldb, &C[2]);
  addDot(k, &A[0], &B[3], ldb, &C[3]);
}

void  gemm_4x4block_v3(float *A, float *B, float *C, int m, int n, int k) {
  int lda = k, ldb = n, ldc = n;
  for (int i = 0; i < m; i += 4) {
    for (int j = 0; j < n; j += 4) {
      addDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}