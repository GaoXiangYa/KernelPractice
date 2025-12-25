#pragma once

#include "util.h"

#define WARP_SIZE 32

__inline__ __device__ float warpReduceMax(float val) {
  auto mask = __activemask();
  for (int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1) {
    val = MAX(val, __shfl_down_sync(mask, val, offset));
  }
  return val;
}

__inline__ __device__ float warpReduceSum(float val) {
  auto mask = __activemask();
  for (int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1) {
    val += __shfl_down_sync(mask, val, offset);
  }
  return val;
}

__inline__ __device__ float blockReduceMax(float val) {
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

__inline__ __device__ float blockReduceSum(float sum) {
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
