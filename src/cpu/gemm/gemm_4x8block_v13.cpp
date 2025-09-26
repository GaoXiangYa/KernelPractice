#include "gemm.h"
#include <algorithm>
#include <immintrin.h>
#include <xmmintrin.h>

typedef union {
  __m256 v;
  float d[8];
} vec_t;

static void addDot4x8(int k, float *A, int lda, float *B, int ldb, float *C,
                      int ldc) {
  vec_t c0_vec, c1_vec, c2_vec, c3_vec;
  c0_vec.v = _mm256_load_ps(&C(0, 0));
  c1_vec.v = _mm256_load_ps(&C(1, 0));
  c2_vec.v = _mm256_load_ps(&C(2, 0));
  c3_vec.v = _mm256_load_ps(&C(3, 0));

  vec_t a0_vec, a1_vec, a2_vec, a3_vec;
  a0_vec.v = _mm256_set1_ps(0.00f);
  a1_vec.v = _mm256_set1_ps(0.00f);
  a2_vec.v = _mm256_set1_ps(0.00f);
  a3_vec.v = _mm256_set1_ps(0.00f);

  float *bp0_ptr = &B(0, 0);
  float *bp1_ptr = &B(0, 1);
  float *bp2_ptr = &B(0, 2);
  float *bp3_ptr = &B(0, 3);
  float *bp4_ptr = &B(0, 4);
  float *bp5_ptr = &B(0, 5);
  float *bp6_ptr = &B(0, 6);
  float *bp7_ptr = &B(0, 7);

  vec_t b_vec;
  b_vec.v = _mm256_set1_ps(0.00f);

  for (int p = 0; p < k; ++p) {
    a0_vec.v = _mm256_set1_ps(A(0, p));
    a1_vec.v = _mm256_set1_ps(A(1, p));
    a2_vec.v = _mm256_set1_ps(A(2, p));
    a3_vec.v = _mm256_set1_ps(A(3, p));

    b_vec.v = _mm256_set_ps(*bp7_ptr, *bp6_ptr, *bp5_ptr, *bp4_ptr, *bp3_ptr,
                            *bp2_ptr, *bp1_ptr, *bp0_ptr);

    c0_vec.v = _mm256_fmadd_ps(a0_vec.v, b_vec.v, c0_vec.v);
    c1_vec.v = _mm256_fmadd_ps(a1_vec.v, b_vec.v, c1_vec.v);
    c2_vec.v = _mm256_fmadd_ps(a2_vec.v, b_vec.v, c2_vec.v);
    c3_vec.v = _mm256_fmadd_ps(a3_vec.v, b_vec.v, c3_vec.v);

    bp0_ptr += ldb;
    bp1_ptr += ldb;
    bp2_ptr += ldb;
    bp3_ptr += ldb;
    bp4_ptr += ldb;
    bp5_ptr += ldb;
    bp6_ptr += ldb;
    bp7_ptr += ldb;
  }

  _mm256_store_ps(&C(0, 0), c0_vec.v);
  _mm256_store_ps(&C(1, 0), c1_vec.v);
  _mm256_store_ps(&C(2, 0), c2_vec.v);
  _mm256_store_ps(&C(3, 0), c3_vec.v);
}

void innerKernel(float *A, float *B, float *C, int m, int n, int k, int lda,
                 int ldb, int ldc) {
  const int MR = 4, NR = 8;
  for (int i = 0; i < m; i += MR) {
    for (int j = 0; j < n; j += NR) {
      addDot4x8(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}

void gemm_4x8block_v13(float *A, float *B, float *C, int m, int n, int k) {
  int lda = k, ldb = n, ldc = n;
  const int MC = 256;
  const int KC = 128;
  for (int p = 0; p < k; p += KC) {
    int pb = std::min(k - p, KC);
    for (int i = 0; i < m; i += MC) {
      int ib = std::min(m - i, MC);
      innerKernel(&A(i, p), &B(p, 0), &C(i, 0), ib, n, pb, lda, ldb, ldc);
    }
  }
}
