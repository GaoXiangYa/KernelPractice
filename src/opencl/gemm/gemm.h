#pragma once

// GEMM: C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]

void gemm_v0(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);
double gemm_v0_benchmark(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);

void gemm_v1(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);
double gemm_v1_benchmark(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);

void gemm_v2(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);
double gemm_v2_benchmark(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);

void gemm_v3(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);
double gemm_v3_benchmark(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);

void gemm_v4(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);
double gemm_v4_benchmark(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);

void gemm_v5(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);
double gemm_v5_benchmark(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);

void gemm_v6(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);
double gemm_v6_benchmark(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);

void gemm_v7(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);

void gemm_v8(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);

void gemm_v9(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);
double gemm_v9_benchmark(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);

void gemm_v10(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);