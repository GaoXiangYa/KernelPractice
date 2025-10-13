#define A(i, j) A[(i) * lda + (j)]
#define B(i, j) B[(i) * ldb + (j)]
#define C(i, j) C[(i) * ldc + (j)]
#define C_ref(i, j) C_ref[(i) * ldc_ref + (j)]

#define X(i) x[(i) * incx]
#define Y(i) y[(i) * incy]

inline void addDot(int k, float *x, float *y, int incy, float *gamma) {
  /* compute gamma := x' * y + gamma with vectors x and y of length n.

     Here y starts at location y with increment (stride) incx and x starts at
     location  and has (implicit) stride of 1.
  */

  for (int p = 0; p < k; p++) {
    *gamma += x[p] * Y(p);
  }
}

void gemm_v0(float *A, float *B, float *C, int m, int n, int k);

void gemm_v1(float *A, float *B, float *C, int m, int n, int k);

void gemm_v2(float *A, float *B, float *C, int m, int n, int k);

void gemm_v3(float *A, float *B, float *C, int m, int n, int k);

void gemm_v4(float *A, float *B, float *C, int m, int n, int k);

void gemm_v5(float *A, float *B, float *C, int m, int n, int k);

void gemm_v6(float *A, float *B, float *C, int m, int n, int k);

void gemm_v7(float *A, float *B, float *C, int m, int n, int k);

void gemm_v8(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x4block_v3(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x4block_v4(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x4block_v5(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x4block_v6(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x4block_v7(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x4block_v8(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x4block_v9(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x4block_v10(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x4block_v11(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x8block_v12(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x8block_v13(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x8block_v14(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x8block_v15(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x8block_v16(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x8block_v17(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x8block_v18(float *A, float *B, float *C, int m, int n, int k);

void gemm_4x8block_v19(float *A, float *B, float *C, int m, int n, int k);

void gemm_blas(float *A, float *B, float *C, int m, int n, int k);
