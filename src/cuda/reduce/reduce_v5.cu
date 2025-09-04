#include <cstddef>
#include <cuda_runtime.h>
#include <vector>
#include "util.h"
#include "benchmark.cuh"

#define WARP_SIZE 32

template <size_t BLOCK_SIZE>
__device__ __forceinline__ float warpReduceSum(float sum) {
  size_t mask = __activemask();
  if constexpr (BLOCK_SIZE >= 32) {
    sum += __shfl_down_sync(mask, sum, 16);
  }
  if constexpr (BLOCK_SIZE >= 16) {
    sum += __shfl_down_sync(mask, sum, 8);
  }
  if constexpr (BLOCK_SIZE >= 8) {
    sum += __shfl_down_sync(mask, sum, 4);
  }
  if constexpr (BLOCK_SIZE >= 4) {
    sum += __shfl_down_sync(mask, sum, 2);
  }
  if constexpr (BLOCK_SIZE >= 2) {
    sum += __shfl_down_sync(mask, sum, 1);
  }
  return sum;
}

template <size_t BLOCK_SIZE, int COARSE_FACTOR>
__global__ void reduce_kernel_v5(const float *input, float *output, int N) {
  const int segment = COARSE_FACTOR * blockDim.x * blockIdx.x;
  const int tx = threadIdx.x;
  const int base = segment + tx;
  float sum = 0.0f;

#pragma unroll
  for (int tile = 0; tile < COARSE_FACTOR; ++tile) {
    int idx = base + tile * BLOCK_SIZE;
    if (idx < N) {
      sum += input[idx];
    }
  }

  static __shared__ float warp_level_sums[WARP_SIZE];
  const int lane_id = tx & (WARP_SIZE - 1);
  const int wrap_id = tx >> 5;

  sum = warpReduceSum<BLOCK_SIZE>(sum);

  if (lane_id == 0) {
    warp_level_sums[wrap_id] = sum;
  }
  __syncthreads();

  sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warp_level_sums[lane_id] : 0;

  // final reduce using first wrap
  if (wrap_id == 0) {
    sum = warpReduceSum<BLOCK_SIZE / WARP_SIZE>(sum);
  }

  if (tx == 0) {
    output[blockIdx.x] = sum;
  }
}

void reduce_v5(float *input, size_t input_count, float *output) {
  size_t input_size = input_count * sizeof(float);
  const int THREAD_COUNT = 32;
  const int COARSE_FACTOR = 4;
  const int BLOCK_COUNT = (input_count + THREAD_COUNT * COARSE_FACTOR - 1) /
                          (THREAD_COUNT * COARSE_FACTOR);

  size_t output_size = BLOCK_COUNT * sizeof(float);

  float *input_dev = nullptr;
  auto err = cudaMalloc(&input_dev, input_size);
  cudaMemcpy(input_dev, input, input_size,
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  float *output_dev = nullptr;
  err = cudaMalloc(&output_dev, output_size);
  cudaMemcpy(output_dev, output, output_size,
             cudaMemcpyKind::cudaMemcpyHostToDevice);
  float *output_host = (float *)std::malloc(output_size);

  reduce_kernel_v5<THREAD_COUNT, COARSE_FACTOR>
      <<<BLOCK_COUNT, THREAD_COUNT>>>(input_dev, output_dev, input_count);

  cudaMemcpy(output_host, output_dev, output_size,
             cudaMemcpyKind::cudaMemcpyDeviceToHost);

  float sum = 0.0f;
  for (int i = 0; i < BLOCK_COUNT; ++i) {
    sum += output_host[i];
  }

  *output = sum;
}

void reduce_v5_benchmark() {
  const size_t count = 32 * 1024 * 1024;
  const size_t input_size = count * sizeof(float);
  const int repeat = 1;

  std::vector<float> input(count, 0.0f);
  init_random(input);
  const int THREAD_COUNT = 64;
  const int COARSE_FACTOR = 4;
  const int BLOCK_COUNT = (count + THREAD_COUNT * COARSE_FACTOR - 1) /
                          (THREAD_COUNT * COARSE_FACTOR);
  std::vector<float> output(BLOCK_COUNT, 0.0f);

  float *input_dev = nullptr;
  auto err = cudaMalloc(&input_dev, input_size);
  float *output_dev = nullptr;
  err = cudaMalloc(&output_dev, BLOCK_COUNT * sizeof(float));

  cudaMemcpy(input_dev, input.data(), input_size,
             cudaMemcpyKind::cudaMemcpyHostToDevice);
  cudaMemcpy(output_dev, output.data(), BLOCK_COUNT * sizeof(float),
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  double flops = 1.0 * count;
  double bytes = 2.0 * input_size;

  benchmarkKernel("reduce_kernel_v5<64, 4>", BLOCK_COUNT, THREAD_COUNT, flops,
                  bytes, repeat, reduce_kernel_v5<THREAD_COUNT, COARSE_FACTOR>,
                  input_dev, output_dev, count);
}