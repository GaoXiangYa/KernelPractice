#pragma once

#define GEMM_A(i, j) A[(i) * lda + (j)]
#define GEMM_B(i, j) B[(i) * ldb + (j)]
#define GEMM_C(i, j) C[(i) * ldc + (j)]

void gemm_v0(const float* a, const float* b, float* c, int M, int N, int K);
void gemm_v1(const float* a, const float* b, float* c, int M, int N, int K);
void gemm_v2(const float* a, const float* b, float* c, int M, int N, int K);
void gemm_v3(const float* a, const float* b, float* c, int M, int N, int K);
void gemm_v4(const float* a, const float* b, float* c, int M, int N, int K);
void gemm_v5(const float* a, const float* b, float* c, int M, int N, int K);
void gemm_v6(const float* a, const float* b, float* c, int M, int N, int K);
void gemm_v7(const float* a, const float* b, float* c, int M, int N, int K);
void gemm_v8(const float* a, const float* b, float* c, int M, int N, int K);
void gemm_v9(const float* a, const float* b, float* c, int M, int N, int K);
void gemm_v10(const float* a, const float* b, float* c, int M, int N, int K);
void gemm_v11(const float* a, const float* b, float* c, int M, int N, int K);
void gemm_v12(const float* a, const float* b, float* c, int M, int N, int K);