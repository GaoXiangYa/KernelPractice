#include "gemv.h"
#include "util.h"

#define WARP_SIZE 32

static __device__ float warpReduceSum(float sum) {
  auto mask = __activemask();
  for (int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1) {
    sum += __shfl_down_sync(mask, sum, offset);
  }
  return sum;
}

// each thread calculate at least 4 rols
// each block calculate at least nPerThreads row
static __global__ void gemv_kernel_v6(const float4 *mat_a, const float4 *vec_x,
                                      float *vec_y, const int m, const int n_4,
                                      const float alpha, const float beta) {
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row_idx = blockIdx.x * blockDim.y + ty;
  if (row_idx >= m) {
    return;
  }
  const float y_val = beta * vec_y[row_idx];

  float sum = 0.0f;
#pragma unroll
  for (int n_idx = tx; n_idx < n_4; n_idx += blockDim.x) {
    float4 x_val = vec_x[n_idx];
    float4 mat_val = mat_a[row_idx * n_4 + n_idx];
    sum += (mat_val.x * x_val.x + mat_val.y * x_val.y + mat_val.z * x_val.z +
            mat_val.w * x_val.w);
  }

  sum = warpReduceSum(sum);
  if (tx == 0) {
    vec_y[row_idx] = alpha * sum + y_val;
  }
}

void gemv_v6(float *mat_a, float *vec_x, float *vec_y, const int m, const int n,
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

  constexpr int BLOCK_SIZE = 32;
  const int SHARED_MEM_SIZE = n * sizeof(float);
  dim3 block(BLOCK_SIZE, 2);
  dim3 grid((m + 2 - 1) / 2);

  gemv_kernel_v6<<<grid, block>>>(
      (const float4 *)dev_mat_a, (const float4 *)dev_vec_x, dev_vec_y, m, n / 4,
      alpha, beta);

  CHECK_CUDA(
      cudaMemcpy(vec_y, dev_vec_y, m * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_mat_a));
  CHECK_CUDA(cudaFree(dev_vec_x));
  CHECK_CUDA(cudaFree(dev_vec_y));
}

void benchmark_gemv_v6(std::ofstream &file, const int m, const int n) {
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
  dim3 block(BLOCK_SIZE, 2);
  dim3 grid((m + 2 - 1) / 2);

  // -----------------------
  // 1. warmup
  // -----------------------

  constexpr int warmup = 10;
  for (int i = 0; i < warmup; ++i) {
    gemv_kernel_v6<<<grid, block>>>(
        (const float4 *)dev_mat_a, (const float4 *)dev_vec_x, dev_vec_y, m,
        n / 4, alpha, beta);
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
    gemv_kernel_v6<<<grid, block>>>(
        (const float4 *)dev_mat_a, (const float4 *)dev_vec_x, dev_vec_y, m,
        n / 4, alpha, beta);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time_ms, start, stop);

  // 平均每次的时间（毫秒）
  time_ms /= iters;

  const float Flops = 2 * m * n;
  double gflops = Flops / (time_ms * 1e6);

  file << "GEMV_V6: M = " << m << " N = " << n << " -> " << time_ms << " ms "
       << gflops << " GFlops\n";

  CHECK_CUDA(cudaFree(dev_mat_a));
  CHECK_CUDA(cudaFree(dev_vec_x));
  CHECK_CUDA(cudaFree(dev_vec_y));
}
