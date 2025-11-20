#include "gemv.h"
#include "util.h"

static __global__ void gemv_kernel_v0(float *mat_a, float *vec_x, float *vec_y,
                                      const int m, const int n,
                                      const float alpha, const float beta) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
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
  dim3 block(BLOCK_SIZE);
  dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE);

  gemv_kernel_v0<<<grid, block>>>(dev_mat_a, dev_vec_x, dev_vec_y, m, n, alpha,
                                  beta);

  CHECK_CUDA(
      cudaMemcpy(vec_y, dev_vec_y, m * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_mat_a));
  CHECK_CUDA(cudaFree(dev_vec_x));
  CHECK_CUDA(cudaFree(dev_vec_y));
}

void benchmark_gemv_v0(const int m, const int n) {
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  std::vector<float> mat_a(m * n, 0.0f);
  std::vector<float> vec_x(n, 0.0f);
  std::vector<float> vec_y(m, 0.0f);

  init_random(mat_a, -1.0f, 1.0f);
  init_random(vec_x, -1.0f, 1.0f);
  init_random(vec_y, -1.0f, 1.0f);

  float *dev_mat_a = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_mat_a, mat_a.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_mat_a, mat_a.data(), m * n * sizeof(float),
                        cudaMemcpyHostToDevice));
  float *dev_vec_x = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_vec_x, m * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_vec_x, vec_x.data(), m * sizeof(float),
                        cudaMemcpyHostToDevice));

  float *dev_vec_y = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_vec_y, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_vec_y, vec_y.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  constexpr int BLOCK_SIZE = 32;
  dim3 block(BLOCK_SIZE);
  dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // -----------------------
  // 1. warmup
  // -----------------------

  constexpr int warmup = 10;
  for (int i = 0; i < warmup; ++i) {
    gemv_kernel_v0<<<grid, block>>>(dev_mat_a, dev_vec_x, dev_vec_y, m, n,
                                    alpha, beta);
  }

  // -----------------------
  // 2. benchmark（用 cudaEvent）
  // -----------------------
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  float time_ms = 0;
  constexpr int iters = 10;
  for (int i = 0; i < iters; i++) {
    gemv_kernel_v0<<<grid, block>>>(dev_mat_a, dev_vec_x, dev_vec_y, m, n,
                                    alpha, beta);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time_ms, start, stop);

  // 平均每次的时间（毫秒）
  time_ms /= iters;

  const float Flops = 2 * m * n;
  double gflops = Flops / (time_ms * 1e6);

  printf("GEMV_V0 : m=%d n=%d -> %.4f ms, %.2f GFLOPS\n", m, n, time_ms,
         gflops);

  CHECK_CUDA(cudaFree(dev_mat_a));
  CHECK_CUDA(cudaFree(dev_vec_x));
  CHECK_CUDA(cudaFree(dev_vec_y));
}
