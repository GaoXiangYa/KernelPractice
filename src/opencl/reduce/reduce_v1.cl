__kernel void reduce_v1_kernel(__global float* input, __global float* output, const int N, const int work_group_size) {
  const int local_size = get_local_size(0);
  const int local_id = get_local_id(0);
  const int group_id = get_group_id(0);
  const int gid = get_global_id(0);
  __local float shared_data[256 + 16]; // 防止bank conflict

  shared_data[local_id] = (gid < N) ? input[gid] : 0.0f;
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