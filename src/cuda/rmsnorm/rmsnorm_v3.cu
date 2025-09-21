#include "util.cuh"
#include "util.h"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

static __device__ float warpReduceSum(float sum) {
  auto mask = __activemask();
  for (int i = warpSize / 2; i > 0; i >>= 1) {
    sum += __shfl_xor_sync(mask, sum, i);
  }
  return sum;
}

__global__ void rmsnorm_v3_kernel(float4 *input, float4 *output, int input_len,
                                  int round, int eps) {
  const int tx = threadIdx.x;
  const int lane_id = tx & (warpSize - 1);
  const int warp_id = tx >> 5;
  float sum = 0.0f;
  const int num_warps = blockDim.x / warpSize;
  extern __shared__ float shmem[];
  float4 *input_s = reinterpret_cast<float4 *>(&shmem[num_warps]);
  float *warp_level_sums = &shmem[0];

  // step1 square
#pragma unroll 4
  for (int r = 0; r < round; ++r) {
    int idx = tx + r * blockDim.x;
    auto val = __ldg(&input[idx]);
    input_s[idx] = val;
    sum += (val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w);
  }
  __syncthreads();

  // step2 warp reduce
  sum = warpReduceSum(sum);
  if (lane_id == 0) {
    warp_level_sums[warp_id] = sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    sum = tx < num_warps ? warp_level_sums[tx] : 0.0f;
    sum = warpReduceSum(sum);
    if (lane_id == 0) {
      warp_level_sums[0] = sum;
    }
  }
  __syncthreads();

  // step3 rms
  float rms = rsqrtf(warp_level_sums[0] / input_len + eps);

  // step4 output rms
  for (int r = 0; r < round; ++r) {
    int idx = tx + r * blockDim.x;
    auto val = input_s[idx];
    float x = val.x / rms, y = val.y / rms, z = val.z / rms, w = val.w / rms;
    output[idx] = {x, y, z, w};
  }
}

void rmsnorm_v3(float *input, float *output, const int input_len,
                const float eps) {
  float4 *input_vec4 =
      reinterpret_cast<float4 *>(std::malloc(input_len / 4 * sizeof(float4)));
  convertToFloat4(input, input_vec4, input_len);

  float4 *output_vec4 =
      reinterpret_cast<float4 *>(std::malloc(input_len / 4 * sizeof(float4)));
  convertToFloat4(output, output_vec4, input_len);

  size_t input_size = input_len / 4 * sizeof(float4);
  size_t output_size = input_len / 4 * sizeof(float4);

  float4 *input_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&input_dev, input_size));
  CHECK_CUDA(cudaMemcpy(input_dev, input_vec4, input_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  float4 *output_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&output_dev, output_size));
  CHECK_CUDA(cudaMemcpy(output_dev, output_vec4, output_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  constexpr int COARSE_FACTOR = 4;
  const int THREAD_COUNT = std::min(1024, input_len / COARSE_FACTOR);

  const int WARP_NUM = THREAD_COUNT / 32;
  const size_t WARP_SIZE = WARP_NUM * sizeof(float);
  int round = (input_len + (THREAD_COUNT * COARSE_FACTOR) - 1) /
              (THREAD_COUNT * COARSE_FACTOR);
  const size_t SHARED_MEM_SIZE = input_size + WARP_NUM * sizeof(float);
  rmsnorm_v3_kernel<<<1, THREAD_COUNT, SHARED_MEM_SIZE>>>(
      input_dev, output_dev, input_len, round, eps);

  cudaMemcpy(output_vec4, output_dev, output_size,
             cudaMemcpyKind::cudaMemcpyDeviceToHost);

  convertToFloat(output_vec4, output, input_len / 4);
}

void rmsnorm_v3_benchmark() {
  std::cout << "rmsborm_v3_benchmark begin: \n";
  const int input_len = 8192;
  std::vector<float> input(input_len, 0.0f);
  std::vector<float> output(input_len, -1.0f);
  std::vector<float> weight(input_len, 1.0f);

  const float eps = 1.0000f;

  float4 *input_vec4 =
      reinterpret_cast<float4 *>(std::malloc(input_len / 4 * sizeof(float4)));
  convertToFloat4(input.data(), input_vec4, input_len);

  float4 *output_vec4 =
      reinterpret_cast<float4 *>(std::malloc(input_len / 4 * sizeof(float4)));
  convertToFloat4(output.data(), output_vec4, input_len);

  size_t input_size = input_len / 4 * sizeof(float4);
  size_t output_size = input_len / 4 * sizeof(float4);

  float4 *input_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&input_dev, input_size));
  CHECK_CUDA(cudaMemcpy(input_dev, input_vec4, input_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  float4 *output_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&output_dev, output_size));
  CHECK_CUDA(cudaMemcpy(output_dev, output_vec4, output_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  constexpr int COARSE_FACTOR = 4;
  const int THREAD_COUNT = std::min(1024, input_len / COARSE_FACTOR);

  const int WARP_NUM = THREAD_COUNT / 32;
  const size_t WARP_SIZE = WARP_NUM * sizeof(float);
  int round = (input_len + (THREAD_COUNT * COARSE_FACTOR) - 1) /
              (THREAD_COUNT * COARSE_FACTOR);
  const size_t SHARED_MEM_SIZE = input_size + WARP_NUM * sizeof(float);

  cudaMemcpy(output_vec4, output_dev, output_size,
             cudaMemcpyKind::cudaMemcpyDeviceToHost);

  convertToFloat(output_vec4, output.data(), input_len / 4);

  double flops = 12 * input_len;
  double bytes = 12 * input_size;
  const int repeat = 1000;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    rmsnorm_v3_kernel<<<1, THREAD_COUNT, SHARED_MEM_SIZE>>>(input_dev, output_dev, input_len,
                                           round, eps);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  convertToFloat(output_vec4, output.data(), input_len / 4);
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

  std::cout << "rmsnorm_v3_benchmark end: \n";
}
