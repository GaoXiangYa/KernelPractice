#pragma once

// SGEMV: y[M] = alpha * A[M, N] * x[N] + beta * y[M]
// A is row-major with leading dimension N.

void sgemv_v0(const float* A, const float* x, float* y, int M, int N,
              float alpha = 1.0f, float beta = 0.0f);
double sgemv_v0_benchmark(const float* A, const float* x, float* y, int M,
                          int N, float alpha = 1.0f, float beta = 0.0f);
void sgemv_v1(const float* A, const float* x, float* y, int M, int N,
              float alpha = 1.0f, float beta = 0.0f);
double sgemv_v1_benchmark(const float* A, const float* x, float* y, int M,
                          int N, float alpha = 1.0f, float beta = 0.0f);
void sgemv_v2(const float* A, const float* x, float* y, int M, int N,
              float alpha = 1.0f, float beta = 0.0f);
double sgemv_v2_benchmark(const float* A, const float* x, float* y, int M,
                          int N, float alpha = 1.0f, float beta = 0.0f);
void sgemv_v3(const float* A, const float* x, float* y, int M, int N,
              float alpha = 1.0f, float beta = 0.0f);
double sgemv_v3_benchmark(const float* A, const float* x, float* y, int M,
                          int N, float alpha = 1.0f, float beta = 0.0f);
void sgemv_v4(const float* A, const float* x, float* y, int M, int N,
              float alpha = 1.0f, float beta = 0.0f);
double sgemv_v4_benchmark(const float* A, const float* x, float* y, int M,
                          int N, float alpha = 1.0f, float beta = 0.0f);
void sgemv_v5(const float* A, const float* x, float* y, int M, int N,
              float alpha = 1.0f, float beta = 0.0f);
double sgemv_v5_benchmark(const float* A, const float* x, float* y, int M,
                          int N, float alpha = 1.0f, float beta = 0.0f);
void sgemv_v6(const float* A, const float* x, float* y, int M, int N,
              float alpha = 1.0f, float beta = 0.0f);
double sgemv_v6_benchmark(const float* A, const float* x, float* y, int M,
                          int N, float alpha = 1.0f, float beta = 0.0f);