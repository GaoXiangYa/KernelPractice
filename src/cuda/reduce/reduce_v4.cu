#include <cstddef>
#include <cuda_runtime.h>
#include "benchmark.cuh"
#include "util.h"

template <size_t BLOCK_SIZE>
__device__ void wrapReduce(volatile float *cache, int tid) {
  if constexpr (BLOCK_SIZE >= 64)
    cache[tid] += cache[tid + 32];
  if constexpr (BLOCK_SIZE >= 32)
    cache[tid] += cache[tid + 16];
  if constexpr (BLOCK_SIZE >= 16)
    cache[tid] += cache[tid + 8];
  if constexpr (BLOCK_SIZE >= 8)
    cache[tid] += cache[tid + 4];
  if constexpr (BLOCK_SIZE >= 4)
    cache[tid] += cache[tid + 2];
  if constexpr (BLOCK_SIZE >= 2)
    cache[tid] += cache[tid + 1];
}

template <size_t BLOCK_SIZE, int COARSE_FACTOR>
__global__ void reduce_kernel_v4(float *input, float *output) {
  __shared__ float shmem[BLOCK_SIZE];

  const int segment = COARSE_FACTOR * blockDim.x * blockIdx.x;
  const int tx = threadIdx.x;
  const int i = segment + tx;
  float sum = input[i];

#pragma unroll
  for (int tile = 1; tile < COARSE_FACTOR; ++tile) {
    sum += input[i + tile * BLOCK_SIZE];
  }
  shmem[tx] = sum;
  __syncthreads();

  if constexpr (BLOCK_SIZE >= 512) {
    if (tx < 256) {
      shmem[tx] += shmem[tx + 256];
    }
    __syncthreads();
  }

  if constexpr (BLOCK_SIZE >= 256) {
    if (tx < 128) {
      shmem[tx] += shmem[tx + 128];
    }
    __syncthreads();
  }

  if constexpr (BLOCK_SIZE >= 128) {
    if (tx < 64) {
      shmem[tx] += shmem[tx + 64];
    }
    __syncthreads();
  }

  if (tx < 32) {
    wrapReduce<BLOCK_SIZE>(shmem, tx);
  }

  if (tx == 0) {
    output[blockIdx.x] = shmem[0];
  }
}

void reduce_v4(float *input, size_t input_count, float *output) {
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

  reduce_kernel_v4<THREAD_COUNT, COARSE_FACTOR>
      <<<BLOCK_COUNT, THREAD_COUNT>>>(input_dev, output_dev);

  cudaMemcpy(output_host, output_dev, output_size,
             cudaMemcpyKind::cudaMemcpyDeviceToHost);

  float sum = 0.0f;
  for (int i = 0; i < BLOCK_COUNT; ++i) {
    sum += output_host[i];
  }

  *output = sum;
}

void reduce_v4_benchmark() {
  const size_t count = 32 * 1024 * 1024;
  const size_t input_size = count * sizeof(float);
  const int repeat = 10;

  std::vector<float> input(count, 0.0f);
  init_random(input);
  const int THREAD_COUNT = 256;
  const int COARSE_FACTOR = 8;
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

  benchmarkKernel("reduce_kernel_v4<256, 8>", BLOCK_COUNT, THREAD_COUNT, flops,
                  bytes, repeat, reduce_kernel_v4<THREAD_COUNT, COARSE_FACTOR>,
                  input_dev, output_dev);
}