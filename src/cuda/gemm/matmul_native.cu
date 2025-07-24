#include "matmul.cuh"

__global__ void matmul_native(const float* A, const float* B, float* C, int M, int N, int K) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (ty < M && tx < N) {
    float c = 0;
    for (int i = 0; i < K; ++ i) {
      c += A[ty * K + i] * B[i * N + tx];
    }
    C[ty * N + tx] = c;
  }
}