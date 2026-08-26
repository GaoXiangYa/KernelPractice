// y[M] = alpha * A[M, N] * x[N] + beta * y[M]
// Naive: one work-item per row, serial dot product over N.
// 1: Global Memory Coalescing, each warp per row
kernel void sgemv_v2_kernel(global const float* A, __constant float* x,
                              global float* y, const int M, const int N,
                              float alpha, float beta) {
  const int group_id = get_group_id(0);
  const int warp_size = get_sub_group_size();
  const int warp_id = get_sub_group_id();
  const int lane_id = get_sub_group_local_id();
  const int g_row = group_id * get_num_sub_groups() + warp_id;
  if (g_row >= M) {
    return;
  }

  float sum = 0.0f;
  for (int i = lane_id; i < N; i += warp_size) {
    sum += A[g_row * N + i] * x[i];
  }
  sum = sub_group_reduce_add(sum);

  if (lane_id == 0) {
    y[g_row] = alpha * sum + beta * y[g_row];
  }
}