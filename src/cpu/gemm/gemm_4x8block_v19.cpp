#include "gemm.h"
#include <algorithm>
#include <cstring>
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

  vec_t a0_cur_vec, a1_cur_vec, a2_cur_vec, a3_cur_vec;
  a0_cur_vec.v = _mm256_set1_ps(0.00f);
  a1_cur_vec.v = _mm256_set1_ps(0.00f);
  a2_cur_vec.v = _mm256_set1_ps(0.00f);
  a3_cur_vec.v = _mm256_set1_ps(0.00f);

  vec_t a0_nex_vec, a1_nex_vec, a2_nex_vec, a3_nex_vec;

  float *a0_ptr = packedA;
  float *a1_ptr = packedA + k;
  float *a2_ptr = packedA + 2 * k;
  float *a3_ptr = packedA + 3 * k;
  a0_nex_vec.v = _mm256_broadcast_ss(a0_ptr + 0);
  a1_nex_vec.v = _mm256_broadcast_ss((a1_ptr + 0));
  a2_nex_vec.v = _mm256_broadcast_ss((a2_ptr + 0));
  a3_nex_vec.v = _mm256_broadcast_ss((a3_ptr + 0));

  vec_t b_cur_vec, b_nex_vec;
  b_cur_vec.v = _mm256_set1_ps(0.00f);
  b_nex_vec.v = _mm256_load_ps(&packedB[0 * ldb]);

  for (int p = 1; p <= k; ++p) {

    a0_cur_vec.v = a0_nex_vec.v;
    a1_cur_vec.v = a1_nex_vec.v;
    a2_cur_vec.v = a2_nex_vec.v;
    a3_cur_vec.v = a3_nex_vec.v;
    b_cur_vec.v = b_nex_vec.v;

    c0_vec.v = _mm256_fmadd_ps(a0_cur_vec.v, b_cur_vec.v, c0_vec.v);
    c1_vec.v = _mm256_fmadd_ps(a1_cur_vec.v, b_cur_vec.v, c1_vec.v);
    c2_vec.v = _mm256_fmadd_ps(a2_cur_vec.v, b_cur_vec.v, c2_vec.v);
    c3_vec.v = _mm256_fmadd_ps(a3_cur_vec.v, b_cur_vec.v, c3_vec.v);

    if (p == k) {
      break;
    }

    a0_nex_vec.v = _mm256_broadcast_ss((a0_ptr + p));
    a1_nex_vec.v = _mm256_broadcast_ss((a1_ptr + p));
    a2_nex_vec.v = _mm256_broadcast_ss((a2_ptr + p));
    a3_nex_vec.v = _mm256_broadcast_ss((a3_ptr + p));

    b_nex_vec.v = _mm256_load_ps(&packedB[p * ldb]);
  }

  _mm256_store_ps(&C(0, 0), c0_vec.v);
  _mm256_store_ps(&C(1, 0), c1_vec.v);
  _mm256_store_ps(&C(2, 0), c2_vec.v);
  _mm256_store_ps(&C(3, 0), c3_vec.v);
}

static void storeVal1x8(const float *src, float *dst, int k) {
  for (int p = 0; p < k; p += 8) {
    auto val = _mm256_load_ps(src + p);
    _mm256_store_ps(dst, val);
    dst += 8;
  }
}

static void packedMatrixA(float *A, float *packedA, int ib, int kb, int lda) {

  for (int i = 0; i < ib; i += 4) {
    storeVal1x8(&A(i, 0), packedA, kb);
    storeVal1x8(&A(i + 1, 0), packedA + kb, kb);
    storeVal1x8(&A(i + 2, 0), packedA + 2 * kb, kb);
    storeVal1x8(&A(i + 3, 0), packedA + 3 * kb, kb);
    packedA += 4 * kb;
  }
}

static void packedMatrixB(float *B, float *packedB, int k, int ldb, int nr) {
  for (int p = 0; p < k; p += 4) {
    const float *bp0 = B + p * ldb;
    const float *bp1 = B + (p + 1) * ldb;
    const float *bp2 = B + (p + 2) * ldb;
    const float *bp3 = B + (p + 3) * ldb;

    storeVal1x8(bp0, packedB, nr);
    storeVal1x8(bp1, packedB + nr, nr);
    storeVal1x8(bp2, packedB + 2 * nr, nr);
    storeVal1x8(bp3, packedB + 3 * nr, nr);
    packedB += 4 * nr;
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

void gemm_4x8block_v19(float *A, float *B, float *C, int m, int n, int k) {
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
