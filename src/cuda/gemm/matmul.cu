#include "matmul.cuh"
#include <cuda_runtime.h>

// a[m, n] * b[n, k] = c[m, k]
__global__ void matmul_native_kernel(const float *a, const float *b, float *c,
                                     int m, int n, int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < k) {
    float sum = 0;
    for (int i = 0; i < n; ++i) {
      sum += a[row * n + i] * b[i * k + col];
    }
    c[row * k + col] = sum;
  }
}

// 使用shared memory进行优化
template <int BLOCK>
__global__ void matmul_kernel_0(float *A, float *B, float *C, int M, int N,
                                int K) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  float *begin_a = A + by * BLOCK * K;
  float *begin_b = B + bx * BLOCK;
  float *end_a = begin_a + K;

  float sum = 0.0f;
  for (auto a_ptr = begin_a, b_ptr = begin_b; a_ptr < end_a;
       a_ptr += BLOCK, b_ptr += BLOCK * N) {
    __shared__ float shared_a[BLOCK][BLOCK];
    __shared__ float shared_b[BLOCK][BLOCK];
    shared_a[ty][tx] = a_ptr[ty * K + tx];
    shared_b[ty][tx] = b_ptr[ty * N + tx];
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK; ++k) {
      sum += shared_a[ty][k] * shared_b[k][tx];
    }

    __syncthreads();
  }

  C[(BLOCK * by + ty) * N + BLOCK * bx + tx] = sum;
}

void matmul_native(const float *a, const float *b, float *c, int n, int m,
                   int k) {}

void matmul_0(const float *a, const float *b, float *c, int m, int n, int k) {}
