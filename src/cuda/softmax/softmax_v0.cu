#include "cutlass/fast_math.h"
#include "softmax.h"
#include "util.h"

#define WARP_SIZE 32

__device__ float warpReduceMax(float val) {
  auto mask = __activemask();
  for (int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1) {
    val = MAX(val, __shfl_down_sync(mask, val, offset));
  }
  return val;
}

__device__ float warpReduceSum(float val) {
  auto mask = __activemask();
  for (int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1) {
    val += __shfl_down_sync(mask, val, offset);
  }
  return val;
}

__device__ float blockReduceMax(float val) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & (WARP_SIZE - 1);
  __shared__ float shmem_max[WARP_SIZE];

  val = warpReduceMax(val);
  if (lane_id == 0) {
    shmem_max[warp_id] = val;
  }
  __syncthreads();

  val = (tid < blockDim.x / WARP_SIZE) ? shmem_max[lane_id] : 0;

  if (warp_id == 0) {
    val = warpReduceMax(val);
  }

  return val;
}

__device__ float blockReduceSum(float sum) {
  const int tx = threadIdx.x;
  const int warp_id = tx >> 5;
  const int lane_id = tx & (WARP_SIZE - 1);

  __shared__ float shmem[WARP_SIZE];
  sum = warpReduceSum(sum);
  if (lane_id == 0) {
    shmem[warp_id] = sum;
  }
  __syncthreads();

  sum = (tx < blockDim.x / WARP_SIZE) ? shmem[lane_id] : 0.0f;
  if (warp_id == 0) {
    sum = warpReduceSum(sum);
  }
  return sum;
}

// native softmax, each block calculate one token
static __global__ void softmax_v0_kernel(const float *input, float *output,
                                         const int n) {
  const int tx = threadIdx.x;
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= n) {
    return;
  }

  float max_num = 0.0f;
  max_num = blockDim.x <= 32 ? warpReduceMax(max_num) : blockReduceMax(max_num);

  __shared__ float s_max, s_sum;
  if (tx == 0) {
    s_max = max_num;
  }
  __syncthreads();

  float tmp = static_cast<float>(__ldg(input + idx));
  float sum_num = cutlass::fast_exp(tmp - max_num);
  sum_num = blockDim.x <= 32 ? warpReduceSum(sum_num) : blockReduceSum(sum_num);

  if (tx == 0) {
    s_sum = sum_num;
  }
  __syncthreads();

  output[idx] = cutlass::fast_exp(tmp - s_max) / s_sum;
}

void softmax_v0(float *input, float *output, int n) {
  float *input_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&input_dev, n * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(input_dev, input, n * sizeof(float), cudaMemcpyHostToDevice));

  float *output_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&output_dev, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(output_dev, output, n * sizeof(float),
                        cudaMemcpyHostToDevice));

  constexpr int kThreadPerBlock = 1024;
  constexpr int kBlockPerGrid = 1;
  dim3 grid(kBlockPerGrid);
  dim3 block(kThreadPerBlock);

  softmax_v0_kernel<<<grid, block>>>(input_dev, output_dev, n);

  CHECK_CUDA(cudaMemcpy(output, output_dev, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
}
