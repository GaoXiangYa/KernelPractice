#define COARSE_FACTOR 8

__kernel void reduce_v2_kernel(__global float* input, __global float* output, const int N) {
  const int local_size = get_local_size(0);
  const int local_id = get_local_id(0);
  const int group_id = get_group_id(0);
  const int gid = get_global_id(0);

  if (gid >= N) return;

  __local float shared_data[256 + 1]; // 防止bank conflict
  float sum = 0.0f;

#pragma unroll
  for (int i = 0; i < COARSE_FACTOR; ++i) {
    int idx = local_id + i * local_size + group_id * local_size * COARSE_FACTOR;
    if (idx < N) {
      sum += input[idx];
    }
  }
  shared_data[local_id] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
    if (local_id < stride) {
      shared_data[local_id] += shared_data[local_id + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0) {
    output[group_id] = shared_data[0];
  }
}