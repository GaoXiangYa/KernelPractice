#include "gemm.h"

static void addDot4x4(int k, float *A, int lda, float *B, int ldb, float *C,
                      int ldc) {
  // addDot(k, &A(0, 0), &B(0, 0), ldb, &C(0, 0));
  for (int p = 0; p < k; ++p) {
    C(0, 0) += A(0, p) * B(p, 0);
    C(0, 1) += A(0, p) * B(p, 1);
    C(0, 2) += A(0, p) * B(p, 2);
    C(0, 3) += A(0, p) * B(p, 3);

    C(1, 0) += A(1, p) * B(p, 0);
    C(1, 1) += A(1, p) * B(p, 1);
    C(1, 2) += A(1, p) * B(p, 2);
    C(1, 3) += A(1, p) * B(p, 3);

    C(2, 0) += A(2, p) * B(p, 0);
    C(2, 1) += A(2, p) * B(p, 1);
    C(2, 2) += A(2, p) * B(p, 2);
    C(2, 3) += A(2, p) * B(p, 3);

    C(3, 0) += A(3, p) * B(p, 0);
    C(3, 1) += A(3, p) * B(p, 1);
    C(3, 2) += A(3, p) * B(p, 2);
    C(3, 3) += A(3, p) * B(p, 3);
  }
}

void gemm_4x4block_v5(float *A, float *B, float *C, int m, int n, int k) {
  int lda = k, ldb = n, ldc = n;
  for (int i = 0; i < m; i += 4) {
    for (int j = 0; j < n; j += 4) {
      addDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}