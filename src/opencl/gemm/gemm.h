#pragma once

// GEMM: C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]

void gemm_v0(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);

void gemm_v1(const float* A, const float* B, float* C, int M, int N, int K, float alpha=1.0f, float beta=0.0f);