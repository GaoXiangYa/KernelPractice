#include "gemm.h"

// A[m, k] * B[k, n] = C[m, n]
void gemm_v0(float* A, float* B, float* C, int m, int n, int k) {
  int lda = k, ldb = n, ldc = n;

  for (int i = 0; i < m; ++ i) {
    for (int j = 0; j < n; ++ j) {
      for (int p = 0; p < k; ++ p) {
        C(i, j) += A(i, p) * B(p, j);
      }
    }
  }
}