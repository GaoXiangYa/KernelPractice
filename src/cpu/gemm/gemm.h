#define A(i, j)     A[(i) * lda     + (j)]
#define B(i, j)     B[(i) * ldb     + (j)]
#define C(i, j)     C[(i) * ldc     + (j)]
#define C_ref(i, j) C_ref[(i) * ldc_ref + (j)]

void gemm_v0(float* A, float* B, float* C, int m, int n, int k);