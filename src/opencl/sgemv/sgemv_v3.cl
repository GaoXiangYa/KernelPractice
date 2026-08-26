// y[M] = alpha * A[M, N] * x[N] + beta * y[M]
// Naive: one work-item per row, serial dot product over N.
// 1: Global Memory Coalescing, each warp per row
// 2: Store vector x to __constant memory
// 3: Vectorized load
kernel void sgemv_v3_kernel(global const float* A, __constant float4* x,
                              global float* y, const int M, const int N,
                              float alpha, float beta) {
  const int group_id = get_group_id(0);
  const int warp_size = get_sub_group_size();
  const int warp_id = get_sub_group_id();
  const int lane_id = get_sub_group_local_id();
  const int g_row = group_id * get_num_sub_groups() + warp_id;
  const int len = (N >> 2);

  if (g_row >= M) {
    return;
  }

  float sum = 0.0f;
  for (int i = lane_id; i < len; i += warp_size) {
    float4 vec_a = *(float4*)(&A[g_row * N + (i << 2)]);
    float4 vec_x = x[i];
    sum += (vec_a.x * vec_x.x + vec_a.y * vec_x.y + vec_a.z * vec_x.z + vec_a.w * vec_x.w);
  }
  sum = sub_group_reduce_add(sum);

  if (lane_id == 0) {
    y[g_row] = alpha * sum + beta * y[g_row];
  }
}