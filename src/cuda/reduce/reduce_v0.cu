#include "benchmark.cuh"
#include "reduce.h"
#include "util.h"
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

__global__ void reduce_kernel_v0(float *input, float *output) {
  const int segment = blockIdx.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = threadIdx.x + segment;
  for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
    if (threadIdx.x < stride) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[blockIdx.x] = input[tid];
  }
}

__host__ void reduce_v0(float *input, size_t input_count, float *output) {
  size_t input_size = input_count * sizeof(float);
  const int THREAD_COUNT = 32;
  const int BLOCK_COUNT = (input_count + THREAD_COUNT - 1) / (THREAD_COUNT);

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

  reduce_kernel_v0<<<BLOCK_COUNT, THREAD_COUNT>>>(input_dev, output_dev);

  cudaMemcpy(output_host, output_dev, output_size,
             cudaMemcpyKind::cudaMemcpyDeviceToHost);

  float sum = 0.0f;
  for (int i = 0; i < BLOCK_COUNT; ++i) {
    sum += output_host[i];
  }

  *output = sum;
}

void reduce_v0_benchmark() {
  const size_t count = 32 * 1024 * 1024;
  const size_t input_size = count * sizeof(float);
  const int repeat = 1;

  std::vector<float> input(count, 0.0f);
  init_random(input);
  const int THREAD_COUNT = 32;
  const int BLOCK_COUNT = (count + THREAD_COUNT - 1) / (THREAD_COUNT);
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
  benchmarkKernel("reduce_kernel_v0", BLOCK_COUNT, THREAD_COUNT, flops, bytes,
                  repeat, reduce_kernel_v0, input_dev, output_dev);
  // BENCHMARK_KERNEL(reduce_kernel_v0, BLOCK_COUNT, THREAD_COUNT, flops, bytes,
  //                repeat, input_dev, output_dev);
}