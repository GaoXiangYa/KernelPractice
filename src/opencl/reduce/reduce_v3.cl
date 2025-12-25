__kernel void reduce_v3_kernel(__global float4* input, __global float4* output, const int N) {
  const int local_size = get_local_size(0);
  const int local_id = get_local_id(0);
  const int group_id = get_group_id(0);
  const int gid = get_global_id(0);

  if (gid >= N / 4) return;

  __local float4 shared_data[256];
  shared_data[local_id] = input[gid];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
    if (local_id < stride) {
      shared_data[local_id].x += shared_data[local_id + stride].x;
      shared_data[local_id].y += shared_data[local_id + stride].y;
      shared_data[local_id].z += shared_data[local_id + stride].z;
      shared_data[local_id].w += shared_data[local_id + stride].w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0) {
    output[group_id] = shared_data[0];
  }
}