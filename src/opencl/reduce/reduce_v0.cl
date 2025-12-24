__kernel void reduce_v0_kernel(__global float* input, __global float* output, const int N) {
    const int local_size = get_local_size(0);
    const int local_id = get_local_id(0);
    const int group_id = get_group_id(0);
    const int gid = get_global_id(0);

    if (gid >= N) return;  // 防止越界

    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (local_id < stride && gid + stride < N) {

            input[gid] += input[gid + stride];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // 每个 work-group 的第 0 个线程写回输出
    if (local_id == 0) {
        output[group_id] += input[gid];
    }
}
