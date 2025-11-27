#include "gemv.h"
#include "util.h"

#define WARP_SIZE 32

static __device__ float warp_reduce(float sum) {
  auto mask = __activemask();
  for (int offset = WARP_SIZE / 2; offset >= 1; offset >>= 1) {
    sum += __shfl_down_sync(mask, sum, offset);
  }
  return sum;
}

template <int THREAD_COUNT>
static __global__ void gemv_kernel_v3(float *mat_a, float *vec_x, float *vec_y,
                                      const int m, const int n,
                                      const float alpha, const float beta) {
  const int row = blockIdx.x;
  const int PER_THREAD_CALCULATE = (n + THREAD_COUNT - 1) / THREAD_COUNT;
  if (row >= m)
    return;
  const int tid = threadIdx.x;
  const float y = beta * vec_y[row];
  const int warp_id = tid >> 5;
  const int lane_id = tid & (WARP_SIZE - 1);
  const int stride = WARP_SIZE * PER_THREAD_CALCULATE;
  const int start_pos = warp_id * stride;
  if (start_pos >= n)
    return;
  extern __shared__ float shmem[];
  __shared__ float warp_level_sums[WARP_SIZE];

  // load vec_x to shared memory
  for (int i = tid; i < n; i += THREAD_COUNT) {
    shmem[i] = vec_x[i];
  }
  __syncthreads();

  // warp level mat_a * vec_x
  float sum = 0.0f;
  int base = lane_id + start_pos;
#pragma unroll
  for (int i = 0; i < PER_THREAD_CALCULATE; ++i) {
    int idx = base + i * WARP_SIZE;
    if (idx < n) {
      sum += mat_a[row * n + idx] * shmem[idx];
    }
  }
  // block level reduce
  // warp internal reduce
  sum = warp_reduce(sum);
  if (lane_id == 0) {
    warp_level_sums[warp_id] = sum;
  }
  __syncthreads();

  // get each warp reduce result
  sum = (tid < blockDim.x / WARP_SIZE) ? warp_level_sums[tid] : 0;
  if (warp_id == 0) {
    // warp level reduce
    sum = warp_reduce(sum);
    if (lane_id == 0) {
      vec_y[row] = alpha * sum + y;
    }
  }
}

void gemv_v3(float *mat_a, float *vec_x, float *vec_y, const int m, const int n,
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

  constexpr int BLOCK_SIZE = 256;
  int PER_THREAD_CALCULATE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int SHARED_MEM_SIZE = n * sizeof(float);
  dim3 block(BLOCK_SIZE);
  dim3 grid(m);
  // dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE);

  gemv_kernel_v3<BLOCK_SIZE><<<grid, block, SHARED_MEM_SIZE>>>(
      dev_mat_a, dev_vec_x, dev_vec_y, m, n, alpha, beta);

  CHECK_CUDA(
      cudaMemcpy(vec_y, dev_vec_y, m * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_mat_a));
  CHECK_CUDA(cudaFree(dev_vec_x));
  CHECK_CUDA(cudaFree(dev_vec_y));
}

void benchmark_gemv_v3(std::ofstream &file, const int m, const int n) {
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

  constexpr int BLOCK_SIZE = 256;
  dim3 block(BLOCK_SIZE);
  dim3 grid(m);

  // -----------------------
  // 1. warmup
  // -----------------------

  constexpr int warmup = 10;
  for (int i = 0; i < warmup; ++i) {
    gemv_kernel_v3<BLOCK_SIZE><<<grid, block, n * sizeof(float)>>>(
        dev_mat_a, dev_vec_x, dev_vec_y, m, n, alpha, beta);
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
    gemv_kernel_v3<BLOCK_SIZE><<<grid, block, n * sizeof(float)>>>(
        dev_mat_a, dev_vec_x, dev_vec_y, m, n, alpha, beta);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time_ms, start, stop);

  // 平均每次的时间（毫秒）
  time_ms /= iters;

  const float Flops = 2 * m * n;
  double gflops = Flops / (time_ms * 1e6);

  file << "GEMV_V3: M = " << m << " N = " << n << " -> " << time_ms << " ms "
       << gflops << " GFlops\n";

  CHECK_CUDA(cudaFree(dev_mat_a));
  CHECK_CUDA(cudaFree(dev_vec_x));
  CHECK_CUDA(cudaFree(dev_vec_y));
}
