// y[M] = alpha * A[M, N] * x[N] + beta * y[M]
// Naive: one work-item per row, serial dot product over N.
// 1: Global Memory Coalescing, each warp per row
// 2: Store vector x to __constant memory
// 3: Vectorized load
// 4: each warp calculate 4 rows
kernel void sgemv_v5_kernel(global const float* A, __constant float* x,
                              global float* y, const int M, const int N,
                              float alpha, float beta) {
  const int group_id = get_group_id(0);
  const int warp_size = get_sub_group_size();
  const int warp_id = get_sub_group_id();
  const int lane_id = get_sub_group_local_id();
  const int g_row_base = (group_id * get_num_sub_groups() + warp_id) << 2;
  const int g_row0 = g_row_base;
  const int g_row1 = g_row_base + 1;
  const int g_row2 = g_row_base + 2;
  const int g_row3 = g_row_base + 3;
  const int warp_stride = warp_size << 2;

  float4 sum = 0.0f;
  float4 vec_a = 0.0f;
  float4 vec_x = 0.0f;
  for (int i = lane_id; i < N; i += warp_stride) {
    vec_a = (float4)(A[g_row0 * N + i], A[g_row0 * N + i + 64], A[g_row0 * N + i + 128], A[g_row0 * N + i + 192]);
    vec_x = (float4)(x[i], x[i + 64], x[i + 128], x[i + 192]);
    sum.x += (vec_a.x * vec_x.x + vec_a.y * vec_x.y + vec_a.z * vec_x.z + vec_a.w * vec_x.w);
    
    vec_a = (float4)(A[g_row1 * N + i], A[g_row1 * N + i + 64], A[g_row1 * N + i + 128], A[g_row1 * N + i + 192]);
    sum.y += (vec_a.x * vec_x.x + vec_a.y * vec_x.y + vec_a.z * vec_x.z + vec_a.w * vec_x.w);
    
    vec_a = (float4)(A[g_row2 * N + i], A[g_row2 * N + i + 64], A[g_row2 * N + i + 128], A[g_row2 * N + i + 192]);
    sum.z += (vec_a.x * vec_x.x + vec_a.y * vec_x.y + vec_a.z * vec_x.z + vec_a.w * vec_x.w);
    
    vec_a = (float4)(A[g_row3 * N + i], A[g_row3 * N + i + 64], A[g_row3 * N + i + 128], A[g_row3 * N + i + 192]);
    sum.w += (vec_a.x * vec_x.x + vec_a.y * vec_x.y + vec_a.z * vec_x.z + vec_a.w * vec_x.w);
  }
  sum.x = sub_group_reduce_add(sum.x);
  sum.y = sub_group_reduce_add(sum.y);
  sum.z = sub_group_reduce_add(sum.z);
  sum.w = sub_group_reduce_add(sum.w);

  if (lane_id == 0) {
    y[g_row0] = alpha * sum.x + beta * y[g_row0];
    y[g_row1] = alpha * sum.y + beta * y[g_row1];
    y[g_row2] = alpha * sum.z + beta * y[g_row2];
    y[g_row3] = alpha * sum.w + beta * y[g_row3];
  }
}