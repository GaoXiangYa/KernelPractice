#define VEC_SIZE 4
#define ROWS_PER_WARP 4
#define WARP_SIZE 64

// ---- Fused RMS Norm: reduce sum(x²) + normalize in one kernel ----
__kernel void rms_norm_fused_v3(
    __global const float4* input,     // [N, D]
    __constant const float4* weight,    // [D]
    __global       float4* output,    // [N, D]
    const int N,
    const int D,
    const float epsilon,
    __local float* tmp_sum) {
    const int lsz = get_local_size(0);
    const int lid = get_local_id(0);
    const int group_id = get_group_id(0);      // one work-group per row of N
    const int sub_group_id = get_sub_group_id();
    const int sub_group_lid = get_sub_group_local_id();
    const int input_len = D / VEC_SIZE;

    // float warp_sum[ROWS_PER_WARP];

    // ---- per-thread partial sum of x² ----
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++ r) {
        float lsum = 0.0f;
        const int row = group_id * ROWS_PER_WARP + r;
        if (row >= N) {
            return;
        }
        for (int i = lid; i < input_len; i += lsz) {
            float4 val = input[row * input_len + i];
            lsum += (val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w);
        }

        float sg_sum = sub_group_reduce_add(lsum);
        if (sub_group_lid == 0) {
            tmp_sum[sub_group_id] = sg_sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // reduce add tmp_sum[0...num_sg - 1] in sub_group0
        if (sub_group_id == 0) {
            const int num_sg = get_num_sub_groups();
            float x = (sub_group_lid < num_sg) ? tmp_sum[sub_group_lid] : 0.0f;
            x = sub_group_reduce_add(x);
            if (sub_group_lid == 0) {
                tmp_sum[0] = x;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        float sum = tmp_sum[0];

        // ---- rms = 1 / sqrt( sum(x²)/D + ε ) ----
        float rms = 1.0f / sqrt(sum / D + epsilon);
        float4 rms4 = {rms, rms, rms, rms};

        // ---- element-wise normalize ----
        for (int i = lid; i < input_len; i += lsz) {
            float4 input_val = input[row * input_len + i];
            float4 weight_val = weight[i];
            output[row * input_len + i] = input_val * weight_val * rms4;
        }
    }
}