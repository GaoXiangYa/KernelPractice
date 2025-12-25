#define WARP_SIZE 32

__kernel void wrapReduce(float4 *cache, int tid)
{
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}

__kernel void reduce_v4_kernel(__global __read_only float4 *input, __global __write_only float4 *output, __local __read_write float4 *local_data, const int N)
{
    const int local_size = get_local_size(0);
    const int local_id = get_local_id(0);
    const int group_id = get_group_id(0);
    const int gid = get_global_id(0);

    if (gid >= N / 4)
        return;
    local_data[local_id] = input[gid];
    for (int stride = local_size >> 1; stride > WARP_SIZE; stride >>= 1)
    {

        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < stride)
        {
            local_data[local_id] += local_data[local_id + stride];
        }
    }

    if (local_id < WARP_SIZE)
    {
      local_data[local_id] += local_data[local_id + 32];
      local_data[local_id] += local_data[local_id + 16];
      local_data[local_id] += local_data[local_id + 8];
      local_data[local_id] += local_data[local_id + 4];
      local_data[local_id] += local_data[local_id + 2];
      local_data[local_id] += local_data[local_id + 1];
    }

    if (local_id == 0)
    {
        output[group_id] = local_data[0];
    }
}