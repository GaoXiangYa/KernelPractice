#include "gemm.h"

static void addDot1x4(int k, float *A, int lda, float *B, int ldb, float *C, int ldc) {
  // addDot(k, &A[0], &B[0], ldb, &C[0]);
  for (int p = 0; p < k; ++p) {
    C(0, 0) += A(0, p) * B(p, 0);
  }

  // addDot(k, &A[0], &B[1], ldb, &C[1]);
  for (int p = 0; p < k; ++p) {
    C(0, 1) += A(0, p) * B(p, 1);
  }

  // addDot(k, &A[0], &B[2], ldb, &C[2]);
  for (int p = 0; p < k; ++p) {
    C(0, 2) += A(0, p) * B(p, 2);
  }

  // addDot(k, &A[0], &B[3], ldb, &C[3]);
  for (int p = 0; p < k; ++ p) {
    C(0, 3) += A(0, p) * B(p, 3);
  }
}

void gemm_v3(float *A, float *B, float *C, int m, int n, int k) {
  int lda = k, ldb = n, ldc = n;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; j += 4) {
      addDot1x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}