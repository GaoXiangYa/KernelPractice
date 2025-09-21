#include "rmsnorm.h"
#include "util.h"
#include <bit>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

template <int SHAEDMEM_SIZE> __device__ void warpReduceSum(float &sum) {
  auto mask = __activemask();
  if constexpr (SHAEDMEM_SIZE >= 32) {
    sum += __shfl_down_sync(mask, sum, 16);
  }
  if constexpr (SHAEDMEM_SIZE >= 16) {
    sum += __shfl_down_sync(mask, sum, 8);
  }
  if constexpr (SHAEDMEM_SIZE >= 8) {
    sum += __shfl_down_sync(mask, sum, 4);
  }
  if constexpr (SHAEDMEM_SIZE >= 4) {
    sum += __shfl_down_sync(mask, sum, 2);
  }
  if constexpr (SHAEDMEM_SIZE >= 2) {
    sum += __shfl_down_sync(mask, sum, 1);
  }
}

template <int SHAEDMEM_SIZE, int COARSE_FACTOR>
__global__ void reduceSum(float *input, float *output, int input_len) {
  __shared__ float input_s[SHAEDMEM_SIZE];
  const int segment = COARSE_FACTOR * blockDim.x * blockIdx.x;
  const int tx = threadIdx.x;
  const int i = tx + segment;

  float sum = 0.0f;
#pragma unroll
  for (int tile = 0; tile < COARSE_FACTOR; ++tile) {
    if ((i + tile * SHAEDMEM_SIZE) < input_len) {
      sum += input[i + tile * SHAEDMEM_SIZE];
    }
  }
  input_s[tx] = sum;

  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tx < stride) {
      input_s[tx] += input_s[tx + stride];
    }
    __syncthreads();
  }

  if (tx < 32) {
    warpReduceSum<SHAEDMEM_SIZE>(input_s[tx]);
  }

  if (tx == 0) {
    atomicAdd(output, input_s[0]);
  }
}

__global__ void square(float *input, float *output, const int input_len) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  auto input_vec4 = std::bit_cast<float4 *>(input);
  auto output_vec4 = std::bit_cast<float4 *>(output);

  if (tid < input_len / 4) {
    float4 in_val = input_vec4[tid];
    float4 out_val = {0.0, 0.0, 0.0, 0.0};

    out_val.x = in_val.x * in_val.x;
    out_val.y = in_val.y * in_val.y;
    out_val.z = in_val.z * in_val.z;
    out_val.w = in_val.w * in_val.w;

    output_vec4[tid] = out_val;
  }

  int base = (input_len / 4) * 4;
  for (int i = base + tid; i < input_len; i += blockDim.x * gridDim.x) {
    output[i] = input[i] * input[i];
  }
}

__global__ void norm(const float *input, float *output, const int input_len,
                     float *rms) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  auto input_vec4 = std::bit_cast<float4 *>(input);
  auto output_vec4 = std::bit_cast<float4 *>(output);

  if (tid < input_len / 4) {
    float4 in_val = input_vec4[tid];
    float4 out_val = {0.0, 0.0, 0.0, 0.0};

    out_val.x = in_val.x / *rms;
    out_val.y = in_val.y / *rms;
    out_val.z = in_val.z / *rms;
    out_val.w = in_val.w / *rms;

    output_vec4[tid] = out_val;
  }

  int base = (input_len / 4) * 4;
  for (int i = base + tid; i < input_len; i += blockDim.x * blockIdx.x) {
    output[i] = input[i] / *rms;
  }
}

__global__ void calculateRms(float *reduce_sum, const int input_len,
                             const float eps, float *rms) {
  *rms = std::sqrt(*reduce_sum / input_len + eps);
}

void rmsnorm_v0(float *input, float *output, const int input_len,
                const float eps) {
  size_t input_size = input_len * sizeof(float);
  size_t output_size = input_len * sizeof(float);

  float *input_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&input_dev, input_size));
  CHECK_CUDA(cudaMemcpy(input_dev, input, input_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  float *output_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&output_dev, output_size));
  CHECK_CUDA(cudaMemcpy(output_dev, output, output_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  // step1 square input
  const int SQUARE_THREAD_COUNT = 32;
  const int SQUARE_BLOCK_COUNT =
      (input_len + SQUARE_THREAD_COUNT - 1) / SQUARE_THREAD_COUNT;
  float *square_output = nullptr;
  CHECK_CUDA(cudaMalloc(&square_output, output_size));
  square<<<SQUARE_BLOCK_COUNT, SQUARE_THREAD_COUNT>>>(input_dev, square_output,
                                                      input_len);

  // step2 calculate rms
  const int THREAD_COUNT = 32;
  const int COARSE_FACTOR = 4;
  const int BLOCK_COUNT = (input_len + THREAD_COUNT * COARSE_FACTOR - 1) /
                          (THREAD_COUNT * COARSE_FACTOR);
  float *reduce_sum_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&reduce_sum_dev, sizeof(float)));

  reduceSum<THREAD_COUNT, COARSE_FACTOR>
      <<<BLOCK_COUNT, THREAD_COUNT>>>(square_output, reduce_sum_dev, input_len);

  float *rms_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&rms_dev, sizeof(float)));
  calculateRms<<<1, 1>>>(reduce_sum_dev, input_len, eps, rms_dev);

  // step3 norm
  const int NORM_THREAD_COUNT = 32;
  const int NORM_BLOCK_COUNT =
      (input_len + NORM_THREAD_COUNT - 1) / NORM_THREAD_COUNT;
  norm<<<NORM_BLOCK_COUNT, NORM_THREAD_COUNT>>>(input_dev, output_dev,
                                                input_len, rms_dev);
  cudaMemcpy(output, output_dev, output_size,
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
}

void rmsnorm_v0_benchmark() {
  std::cout << "rmsnorm_v0_benchmark begin: \n";
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int input_len = 32 * 1024 * 1024;
  std::vector<float> input(input_len);
  std::vector<float> output(input_len);

  init_random(input);

  size_t input_size = input_len * sizeof(float);
  size_t output_size = input_len * sizeof(float);

  float *input_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&input_dev, input_size));
  CHECK_CUDA(cudaMemcpy(input_dev, input.data(), input_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  float *output_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&output_dev, output_size));
  CHECK_CUDA(cudaMemcpy(output_dev, output.data(), output_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  const int SQUARE_THREAD_COUNT = 32;
  const int SQUARE_BLOCK_COUNT =
      (input_len + SQUARE_THREAD_COUNT - 1) / SQUARE_THREAD_COUNT;
  float *square_output = nullptr;
  CHECK_CUDA(cudaMalloc(&square_output, output_size));

  const int THREAD_COUNT = 32;
  const int COARSE_FACTOR = 4;
  const int BLOCK_COUNT = (input_len + THREAD_COUNT * COARSE_FACTOR - 1) /
                          (THREAD_COUNT * COARSE_FACTOR);

  float *reduce_sum_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&reduce_sum_dev, sizeof(float)));
  float *rms_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&rms_dev, sizeof(float)));

  const int NORM_THREAD_COUNT = 32;
  const int NORM_BLOCK_COUNT =
      (input_len + NORM_THREAD_COUNT - 1) / NORM_THREAD_COUNT;
  const float eps = 1.00f;

  int repeat = 1;
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    // step1 square input
    square<<<SQUARE_BLOCK_COUNT, SQUARE_THREAD_COUNT>>>(
        input_dev, square_output, input_len);

    // step2 calculate rms
    reduceSum<THREAD_COUNT, COARSE_FACTOR><<<BLOCK_COUNT, THREAD_COUNT>>>(
        square_output, reduce_sum_dev, input_len);

    calculateRms<<<1, 1>>>(reduce_sum_dev, input_len, eps, rms_dev);

    // step3 norm
    norm<<<NORM_BLOCK_COUNT, NORM_THREAD_COUNT>>>(input_dev, output_dev,
                                                  input_len, rms_dev);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  elapsed_ms /= repeat;  // 平均单次 ms
  elapsed_ms /= 1000.0f; // 转换成秒

  double bytes = 5.0 * input_len * sizeof(float);
  double bandwidth = bytes / elapsed_ms / 1e9;

  double flops = 3*input_len;
  double gflops = flops / elapsed_ms / 1e9;

  printf("Input size: %d\n", input_len);
  printf("Avg Time: %.6f s\n", elapsed_ms);
  printf("Bandwidth: %.2f GB/s\n", bandwidth);
  printf("FLOPS: %.2f GFLOPS\n", gflops);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::cout << "rmsnorm_v0_benchmark end\n";
}