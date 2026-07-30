#pragma once
#include <cstddef>
#include <vector>

// GEMM-Q4_1:  C[M,N] = dequant(A_q4_1) × B[K,N]
//
//   A         [M, rows_per_block]  — Q4_1 layout
//   B         [K, N]               — regular float
//   C         [M, N]               — output
//   block_k                       — quantization group size (typically 32)
//   scale/bias are embedded in A's blocks

void launch_gemm_q4_1_v0(const unsigned char* A, const float* B, float* C,
                         int M, int N, int K, int block_k, float alpha = 1.0f,
                         float beta = 0.0f);
void launch_gemm_q4_1_v1(const unsigned char* A, const float* B, float* C,
                         int M, int N, int K, int block_k, float alpha = 1.0f,
                         float beta = 0.0f);