#include "gemm.h"
#include <immintrin.h>
#include <xmmintrin.h>

typedef union {
  __m128 v;
  float d[4];
} vec_t;

static void addDot4x4(int k, float *A, int lda, float *B, int ldb, float *C,
                      int ldc) {
  vec_t c0_vec, c1_vec, c2_vec, c3_vec;
  c0_vec.v = _mm_set_ps1(0.00f);
  c1_vec.v = _mm_set_ps1(0.00f);
  c2_vec.v = _mm_set_ps1(0.00f);
  c3_vec.v = _mm_set_ps1(0.00f);

  vec_t a0_vec, a1_vec, a2_vec, a3_vec;
  a0_vec.v = _mm_set_ps1(0.00f);
  a1_vec.v = _mm_set_ps1(0.00f);
  a2_vec.v = _mm_set_ps1(0.00f);
  a3_vec.v = _mm_set_ps1(0.00f);

  float *bp0_ptr = &B(0, 0);
  float *bp1_ptr = &B(0, 1);
  float *bp2_ptr = &B(0, 2);
  float *bp3_ptr = &B(0, 3);

  vec_t b_vec;
  b_vec.v = _mm_set_ps1(0.00f);

  for (int p = 0; p < k; ++p) {
    a0_vec.v = _mm_set_ps1(A(0, p));
    a1_vec.v = _mm_set_ps1(A(1, p));
    a2_vec.v = _mm_set_ps1(A(2, p));
    a3_vec.v = _mm_set_ps1(A(3, p));

    b_vec.v = _mm_set_ps(*bp3_ptr, *bp2_ptr, *bp1_ptr, *bp0_ptr);

    c0_vec.v += _mm_mul_ps(a0_vec.v, b_vec.v);
    c1_vec.v += _mm_mul_ps(a1_vec.v, b_vec.v);
    c2_vec.v += _mm_mul_ps(a2_vec.v, b_vec.v);
    c3_vec.v += _mm_mul_ps(a3_vec.v, b_vec.v);

    bp0_ptr += ldb;
    bp1_ptr += ldb;
    bp2_ptr += ldb;
    bp3_ptr += ldb;
  }

  C(0, 0) += c0_vec.d[0];
  C(0, 1) += c0_vec.d[1];
  C(0, 2) += c0_vec.d[2];
  C(0, 3) += c0_vec.d[3];

  C(1, 0) += c1_vec.d[0];
  C(1, 1) += c1_vec.d[1];
  C(1, 2) += c1_vec.d[2];
  C(1, 3) += c1_vec.d[3];

  C(2, 0) += c2_vec.d[0];
  C(2, 1) += c2_vec.d[1];
  C(2, 2) += c2_vec.d[2];
  C(2, 3) += c2_vec.d[3];

  C(3, 0) += c3_vec.d[0];
  C(3, 1) += c3_vec.d[1];
  C(3, 2) += c3_vec.d[2];
  C(3, 3) += c3_vec.d[3];
}

void gemm_4x4block_v10(float *A, float *B, float *C, int m, int n, int k) {
  int lda = k, ldb = n, ldc = n;
  for (int i = 0; i < m; i += 4) {
    for (int j = 0; j < n; j += 4) {
      addDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}