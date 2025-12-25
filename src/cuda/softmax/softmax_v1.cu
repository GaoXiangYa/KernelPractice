#include "reduce.cuh"
#include "softmax.h"
#include "util.h"

#define WARP_SIZE 32

static __global__ void softmax_v1_kernel(const float *input, float *output,
                                         const int n) {
  const int idx = threadIdx.x << 2;

  if (idx >= n) {
    return;
  }

  float max_num = 0.0f;
  max_num = blockDim.x <= 32 ? warpReduceMax(max_num) : blockReduceMax(max_num);

  __shared__ float s_max, s_sum;
  if (idx == 0) {
    s_max = max_num;
  }
  __syncthreads();

  float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};
  tmp.x = static_cast<float>(__ldg(input + idx));
  tmp.y = static_cast<float>(__ldg(input + idx + 1));
  tmp.z = static_cast<float>(__ldg(input + idx + 2));
  tmp.w = static_cast<float>(__ldg(input + idx + 3));

  float sum = 0.0f;
  sum += __expf(tmp.x - s_max);
  sum += __expf(tmp.y - s_max);
  sum += __expf(tmp.z - s_max);
  sum += __expf(tmp.w - s_max);
  sum = blockDim.x <= 32 ? warpReduceSum(sum) : blockReduceSum(sum);

  if (idx == 0) {
    s_sum = sum;
  }
  __syncthreads();

  output[idx] = __expf(tmp.x - s_max) / s_sum;
  output[idx + 1] = __expf(tmp.y - s_max) / s_sum;
  output[idx + 2] = __expf(tmp.z - s_max) / s_sum;
  output[idx + 3] = __expf(tmp.w - s_max) / s_sum;
}

void softmax_v1(float *input, float *output, int n) {
  float *input_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&input_dev, n * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(input_dev, input, n * sizeof(float), cudaMemcpyHostToDevice));

  float *output_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&output_dev, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(output_dev, output, n * sizeof(float),
                        cudaMemcpyHostToDevice));

  constexpr int kThreadPerBlock = 256;
  constexpr int kBlockPerGrid = 1;
  dim3 grid(kBlockPerGrid);
  dim3 block(kThreadPerBlock);

  softmax_v1_kernel<<<grid, block>>>(input_dev, output_dev, n);

  CHECK_CUDA(cudaMemcpy(output, output_dev, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
}
