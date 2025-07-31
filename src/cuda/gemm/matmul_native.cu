#include "matmul.cuh"

// a[m, n] * b[n, k] = c[m, k]
__global__ void matmul_native(const float *a, const float *b, float *c, int m,
                              int n, int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < k) {
    float sum = 0;
    for (int i = 0; i < n; ++ i) {
      sum += a[row * n + i] * b[i * k + col];
    }
    c[row * k + col] = sum;
  }
}
