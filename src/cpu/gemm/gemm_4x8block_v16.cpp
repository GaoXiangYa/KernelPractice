#include "gemm.h"
#include <algorithm>
#include <immintrin.h>
#include <xmmintrin.h>

typedef union {
  __m256 v;
  float d[8];
} vec_t;

static void addDot4x8(int k, float *packedA, float *packedB, int ldb, float *C,
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

  float* a0_ptr = packedA;
  float* a1_ptr = packedA + k;
  float* a2_ptr = packedA + 2 * k;
  float* a3_ptr = packedA + 3 * k;

  vec_t b_vec;
  b_vec.v = _mm256_set1_ps(0.00f);

  for (int p = 0; p < k; ++p) {

    a0_vec.v = _mm256_set1_ps(*(a0_ptr + p));
    a1_vec.v = _mm256_set1_ps(*(a1_ptr + p));
    a2_vec.v = _mm256_set1_ps(*(a2_ptr + p));
    a3_vec.v = _mm256_set1_ps(*(a3_ptr + p));

    b_vec.v = _mm256_load_ps(&packedB[p * ldb]);

    c0_vec.v = _mm256_fmadd_ps(a0_vec.v, b_vec.v, c0_vec.v);
    c1_vec.v = _mm256_fmadd_ps(a1_vec.v, b_vec.v, c1_vec.v);
    c2_vec.v = _mm256_fmadd_ps(a2_vec.v, b_vec.v, c2_vec.v);
    c3_vec.v = _mm256_fmadd_ps(a3_vec.v, b_vec.v, c3_vec.v);
  }

  _mm256_store_ps(&C(0, 0), c0_vec.v);
  _mm256_store_ps(&C(1, 0), c1_vec.v);
  _mm256_store_ps(&C(2, 0), c2_vec.v);
  _mm256_store_ps(&C(3, 0), c3_vec.v);
}

// packed matrix a(mc, kc)
// [a00, a01, a02, a03]
// [a10, a11, a12, a13]
// [a20, a21, a22, a23]
// to:
// (kc, mc)
// [a00, a10, a20]
// [a01, a11, a21]
// [a02, a12, a22]
// [a03, a13, a23]
static void packedMatrixA(float *A, float *packedA, int ib, int kb, int lda) {

  for (int i = 0; i < ib; ++i) {
    for (int j = 0; j < kb; ++j) {
      *packedA++ = A(i, j);
    }
  }
}

static void packedMatrixB(float *B, float *packedB, int k, int ldb, int nr) {
  for (int p = 0; p < k; ++p) {
    const float *bp = B + p * ldb;
    for (int j = 0; j < nr; ++j) {
      *packedB++ = bp[j];
    }
  }
}

static void innerKernel(float *packedA, float *B, float *C, int m, int n, int k,
                        int ldb, int ldc) {
  const int MR = 4, NR = 8;
  alignas(32) float packedB[NR * k];

  for (int j = 0; j < n; j += NR) {
    packedMatrixB(&B(0, j), packedB, k, ldb, NR);
    for (int i = 0; i < m; i += MR) {
      addDot4x8(k, packedA + i * k, packedB, NR, &C(i, j), ldc);
    }
  }
}

void gemm_4x8block_v16(float *A, float *B, float *C, int m, int n, int k) {
  int lda = k, ldb = n, ldc = n;
  const int MC = 256;
  const int NC = 256;
  const int KC = 128;
  alignas(32) float packedA[MC * KC];

  for (int j = 0; j < n; j += NC) {
    int jb = std::min(n - j, NC);
    for (int p = 0; p < k; p += KC) {
      int pb = std::min(k - p, KC);
      for (int i = 0; i < m; i += MC) {
        int ib = std::min(m - i, MC);
        packedMatrixA(&A(i, p), packedA, ib, pb, lda);
        innerKernel(packedA, &B(p, j), &C(i, j), ib, jb, pb, ldb, ldc);
      }
    }
  }
}
