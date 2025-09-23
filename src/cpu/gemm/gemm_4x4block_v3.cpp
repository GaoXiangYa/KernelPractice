#include "gemm.h"

static void addDot4x4(int k, float *A, int lda, float *B, int ldb, float *C, int ldc) {
  addDot(k, &A(0, 0), &B(0, 0), ldb, &C(0, 0));
  addDot(k, &A(0, 0), &B(0, 1), ldb, &C(0, 1));
  addDot(k, &A(0, 0), &B(0, 2), ldb, &C(0, 2));
  addDot(k, &A(0, 0), &B(0, 3), ldb, &C(0, 3));

  addDot(k, &A(1, 0), &B(0, 0), ldb, &C(1, 0));
  addDot(k, &A(1, 0), &B(0, 1), ldb, &C(1, 1));
  addDot(k, &A(1, 0), &B(0, 2), ldb, &C(1, 2));
  addDot(k, &A(1, 0), &B(0, 3), ldb, &C(1, 3));

  addDot(k, &A(2, 0), &B(0, 0), ldb, &C(2, 0));
  addDot(k, &A(2, 0), &B(0, 1), ldb, &C(2, 1));
  addDot(k, &A(2, 0), &B(0, 2), ldb, &C(2, 2));
  addDot(k, &A(2, 0), &B(0, 3), ldb, &C(2, 3));

  addDot(k, &A(3, 0), &B(0, 0), ldb, &C(3, 0));
  addDot(k, &A(3, 0), &B(0, 1), ldb, &C(3, 1));
  addDot(k, &A(3, 0), &B(0, 2), ldb, &C(3, 2));
  addDot(k, &A(3, 0), &B(0, 3), ldb, &C(3, 3));
}

void  gemm_4x4block_v3(float *A, float *B, float *C, int m, int n, int k) {
  int lda = k, ldb = n, ldc = n;
  for (int i = 0; i < m; i += 4) {
    for (int j = 0; j < n; j += 4) {
      addDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}