__kernel void reduce_v0_kernel(__global const float* input, __global float* output, const int n) {
    // Get the global thread ID
    int gid = get_global_id(0);
    
    // Initialize a local sum variable
    float sum = 0.0f;

    // Each thread processes multiple elements
    for (int i = gid; i < n; i += get_global_size(0)) {
        sum += input[i];
    }

    // Use atomic addition to accumulate results in output[0]
    atomic_add(&output[0], sum);
}