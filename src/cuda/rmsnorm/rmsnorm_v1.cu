#include "util.h"
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>

__device__ float warpReduceSum(float sum) {
  auto mask = __activemask();
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(mask, sum, offset);
  }
  return sum;
}

template <int COARSE_FACTOR>
__global__ void rmsnorm_v1_kernel(const float *input, float *output,
                                  const int round, const int input_len,
                                  const float eps) {
  const int tx = threadIdx.x;
  const int lane_id = tx & (warpSize - 1);
  const int warp_id = tx >> 5;
  const int segment = warpSize * COARSE_FACTOR * warp_id;
  const int thread_num = blockDim.x;
  const int warp_num = blockDim.x / warpSize;
  float sum = 0.0f;
  const int stride = blockDim.x * COARSE_FACTOR;
  int base_idx = segment + lane_id;
  extern __shared__ float warp_level_sums[];

  for (int r = 0; r < round; ++r) {
#pragma unroll
    for (int tile = 0; tile < COARSE_FACTOR; ++tile) {
      int idx = base_idx + tile * warpSize + r * stride;
      if (idx < input_len) {
        float v = input[idx];
        sum += v * v;
      }
    }
  }
  // step2 reduce
  sum = warpReduceSum(sum);
  if (lane_id == 0) {
    warp_level_sums[warp_id] = sum;
  }
  __syncthreads();

  sum = tx < warp_num ? warp_level_sums[tx] : 0.0f;
  if (warp_id == 0) {
    sum = warpReduceSum(sum);
    if (lane_id == 0) {
      warp_level_sums[0] = sum;
    }
  }
  __syncthreads();

  // step3 calculate rms
  float rms = std::sqrt(warp_level_sums[0] / input_len + eps);

  // step4 calculate output
  for (int r = 0; r < round; ++r) {
#pragma unroll
    for (int tile = 0; tile < COARSE_FACTOR; ++tile) {
      int idx = base_idx + tile * warpSize + r * stride;
      if (idx < input_len) {
        output[idx] = input[idx] / rms;
      }
    }
  }
}

void rmsnorm_v1(float *input, float *output, const int input_len,
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

  constexpr int COARSE_FACTOR = 4;
  const int THREAD_COUNT = std::min(1024, input_len / COARSE_FACTOR);

  const int WARP_NUM = THREAD_COUNT / 32;
  const size_t WARP_SIZE = WARP_NUM * sizeof(float);
  int round = (input_len + (THREAD_COUNT * COARSE_FACTOR) - 1) /
              (THREAD_COUNT * COARSE_FACTOR);
  rmsnorm_v1_kernel<COARSE_FACTOR><<<1, THREAD_COUNT, WARP_SIZE>>>(
      input_dev, output_dev, round, input_len, eps);

  cudaMemcpy(output, output_dev, output_size,
             cudaMemcpyKind::cudaMemcpyDeviceToHost);
}

void rmsnorm_v1_benchmark() {
  std::cout << "rmsborm_v1_benchmark begin: \n";
  const int input_len = 32 * 1024 * 1024;
  std::vector<float> input(input_len, 0.0f);
  std::vector<float> output(input_len, -1.0f);
  std::vector<float> weight(input_len, 1.0f);

  const float eps = 1.0000f;
  size_t input_size = input_len * sizeof(float);
  size_t output_size = input_len * sizeof(float);

  float *input_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&input_dev, input_size));
  CHECK_CUDA(cudaMemcpy(input_dev, input.data(), input_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  float *weight_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&weight_dev, input_size));
  CHECK_CUDA(cudaMemcpy(weight_dev, weight.data(), input_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  float *output_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&output_dev, output_size));
  CHECK_CUDA(cudaMemcpy(output_dev, output.data(), output_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  double flops = 3 * input_len + 1023 + 3;
  double bytes = 12 * input_size;
  const int repeat = 100;
  constexpr int COARSE_FACTOR = 4;
  const int THREAD_COUNT = std::min(1024, input_len / COARSE_FACTOR);

  const int WARP_NUM = THREAD_COUNT / 32;
  const size_t WARP_SIZE = WARP_NUM * sizeof(float);
  int round = (input_len + (THREAD_COUNT * COARSE_FACTOR) - 1) /
              (THREAD_COUNT * COARSE_FACTOR);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    rmsnorm_v1_kernel<COARSE_FACTOR><<<1, THREAD_COUNT, WARP_SIZE>>>(
        input_dev, output_dev, round, input_len, eps);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  elapsed_ms /= repeat;  // 平均单次 ms
  elapsed_ms /= 1000.0f; // 转换成秒

  double bandwidth = bytes / elapsed_ms / 1e9;
  double gflops = flops / elapsed_ms / 1e9;

  printf("Input size: %d\n", input_len);
  printf("Avg Time: %.6f s\n", elapsed_ms);
  printf("Bandwidth: %.2f GB/s\n", bandwidth);
  printf("FLOPS: %.2f GFLOPS\n", gflops);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::cout << "rmsnorm_v1_benchmark end: \n";
}