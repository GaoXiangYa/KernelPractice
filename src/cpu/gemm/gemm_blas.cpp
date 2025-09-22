#include <cblas.h>
#include "gemm.h"

void gemm_blas(float *A, float *B, float *C, int m, int n, int k) {
  int lda = k, ldb = n, ldc = n;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.00f, A, lda,
              B, ldb, 1.00f, C, ldc);
}