#include "reduce.h"
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void reduce_kernel_v0(float *input, float *output) {
  const int i = 2 * threadIdx.x;
  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    if (threadIdx.x % stride == 0) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[blockIdx.x] = input[0];
  }
}

__host__ void reduce_v0(float *input, size_t input_count, float *output) {
  size_t input_size = input_count * sizeof(float);
  const int THREAD_COUNT = 32;
  const int elements_per_block = 2 * THREAD_COUNT;
  const int BLOCK_COUNT = (input_count + elements_per_block- 1) / elements_per_block;

  size_t output_size = BLOCK_COUNT * sizeof(float);

  float *input_dev = nullptr;
  auto err = cudaMalloc(&input_dev, input_size);
  cudaMemcpy(input_dev, input, input_size,
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  float *output_dev = nullptr;
  err = cudaMalloc(&output_dev, output_size);
  cudaMemcpy(output_dev, output, output_size,
             cudaMemcpyKind::cudaMemcpyHostToDevice);
  float* output_host = (float*)malloc(output_size);

  reduce_kernel_v0<<<BLOCK_COUNT, THREAD_COUNT>>>(input_dev, output_dev);

  cudaMemcpy(output_host, output_dev, output_size,
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
  float sum = 0.0f;
  for (int i = 0; i < BLOCK_COUNT; ++ i) {
    sum += output_host[i];
  }

  *output = sum;
}