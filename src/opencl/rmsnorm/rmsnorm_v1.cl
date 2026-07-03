#define VEC_SIZE 4

// ---- Fused RMS Norm: reduce sum(x²) + normalize in one kernel ----
__kernel void rms_norm_fused_v1(
    __global const float4* input,     // [N, D]
    __constant const float4* weight,    // [D]
    __global       float4* output,    // [N, D]
    const int D,
    const float epsilon) {
    const int lsz = get_local_size(0);
    const int lid = get_local_id(0);
    const int row = get_group_id(0);      // one work-group per row of N
    const int input_len = D / VEC_SIZE;

    // ---- per-thread partial sum of x² ----
    float lsum = 0.0f;
    for (int i = lid; i < input_len; i += lsz) {
        float4 val = input[row * input_len + i];
        lsum += (val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w);
    }
    float sum = sub_group_reduce_add(lsum);

    // ---- rms = 1 / sqrt( sum(x²)/D + ε ) ----
    float rms = 1.0f / sqrt(sum / D + epsilon);
    float4 rms4 = {rms, rms, rms, rms};

    // ---- element-wise normalize ----
    for (int i = lid; i < input_len; i += lsz) {
        output[row * input_len + i] = (input[row * input_len + i] * weight[i] * rms4);
    }
}