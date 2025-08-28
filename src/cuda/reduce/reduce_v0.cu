#include "reduce.h"
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>

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
  float *output_host = (float *)malloc(output_size);

  reduce_kernel_v0<<<BLOCK_COUNT, THREAD_COUNT>>>(input_dev, output_dev);

  cudaMemcpy(output_host, output_dev, output_size,
             cudaMemcpyKind::cudaMemcpyDeviceToHost);

  float sum = 0.0f;
  for (int i = 0; i < BLOCK_COUNT; ++i) {
    sum += output_host[i];
  }

  *output = sum;
}