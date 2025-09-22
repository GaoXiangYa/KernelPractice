#include "gemm.h"

void gemm_v1(float *A, float *B, float *C, int m, int n, int k) {
  // A[m, k] * B[k, n] = C[m, n]
  int lda = k, ldb = n, ldc = n;

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      addDot(k, &A(i, 0), &B(0, j), ldb, &C(i, j));
    }
  }
}