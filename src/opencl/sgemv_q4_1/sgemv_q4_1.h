#pragma once

// SGEMV-Q4_1: y[M] = alpha * dequant(A_q4_1)[M,N] * x[N] + beta * y[M]
//
// A is stored row-major in Q4_1 blocks, one block per `block_k` consecutive
// columns (same layout as gemm_q4_1):
//   [d: float][m: float][qs: block_k/2 bytes of 4-bit nibbles]
// Dequant: a[col] = nibble(col) * d + m, where even columns use the low
// nibble of qs[col/2] and odd columns the high nibble.
//
//   A_q4   [M, (N / block_k) * (8 + block_k / 2)] bytes
//   x      [N]   — regular float
//   y      [M]   — output

void sgemv_q4_1_v0(const unsigned char* A_q4, const float* x, float* y, int M,
                   int N, int block_k, float alpha = 1.0f, float beta = 0.0f);
double sgemv_q4_1_v0_benchmark(const unsigned char* A_q4, const float* x,
                               float* y, int M, int N, int block_k,
                               float alpha = 1.0f, float beta = 0.0f);
