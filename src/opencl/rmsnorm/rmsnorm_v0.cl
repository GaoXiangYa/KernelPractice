// ---- Fused RMS Norm: reduce sum(x²) + normalize in one kernel ----
__kernel void rms_norm_fused_v0(
    __global const float* input,     // [N, D]
    __global const float* weight,    // [D]
    __global       float* output,    // [N, D]
    const int D,
    const float epsilon) {
    const int lsz = get_local_size(0);
    const int lid = get_local_id(0);
    const int row = get_group_id(0);      // one work-group per row of N

    // ---- per-thread partial sum of x² ----
    float lsum = 0.0f;
    for (int i = lid; i < D; i += lsz) {
        float val = input[row * D + i];
        lsum += val * val;
    }
    float sum = sub_group_reduce_add(lsum);

    // ---- rms = 1 / sqrt( sum(x²)/D + ε ) ----
    float rms = 1.0f / sqrt(sum / D + epsilon);

    // ---- element-wise normalize ----
    for (int i = lid; i < D; i += lsz) {
        output[row * D + i] = input[row * D + i] * weight[i] * rms;
    }
}
