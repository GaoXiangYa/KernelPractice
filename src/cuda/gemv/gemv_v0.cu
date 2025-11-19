#include "gemv.h"
#include "util.h"

static __global__ void gemv_kernel_v0(float *mat_a, float *vec_x, float *vec_y,
                                      const int m, const int n,
                                      const float alpha, const float beta) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  if (row >= m) {
    return;
  }

  float sum = 0.0f;
  float y = vec_y[row] * beta;
  for (int col = 0; col < n; ++col) {
    sum += mat_a[row * n + col] * vec_x[col];
  }
  vec_y[row] = alpha * sum + y;
}

void gemv_v0(float *mat_a, float *vec_x, float *vec_y, const int m, const int n,
             const float alpha, const float beta) {
  float *dev_mat_a = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_mat_a, m * n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_mat_a, mat_a, m * n * sizeof(float),
                        cudaMemcpyHostToDevice));

  float *dev_vec_x = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_vec_x, n * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(dev_vec_x, vec_x, n * sizeof(float), cudaMemcpyHostToDevice));

  float *dev_vec_y = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_vec_y, m * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(dev_vec_y, vec_y, m * sizeof(float), cudaMemcpyHostToDevice));

  const int BLOCK_SIZE = 32;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

  gemv_kernel_v0<<<grid, block>>>(dev_mat_a, dev_vec_x, dev_vec_y, m, n, alpha,
                                  beta);

  CHECK_CUDA(
      cudaMemcpy(vec_y, dev_vec_y, m * sizeof(float), cudaMemcpyDeviceToHost));
}
