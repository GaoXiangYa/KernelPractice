// y[M] = alpha * A[M, N] * x[N] + beta * y[M]
// Naive: one work-item per row, serial dot product over N.
// 1: Global Memory Coalescing, each warp per row
// 2: Store vector x to __constant memory
// 3: Vectorized load
// 4: load x to shared memory

#define WARP_SIZE 64
#define MAX_N 16384

kernel void sgemv_v6_kernel(global const float* A, global float* x,
                              global float* y, const int M, const int N,
                              float alpha, float beta) {
  const int group_id = get_group_id(0);
  const int warp_size = get_sub_group_size();
  const int warp_id = get_sub_group_id();
  const int lane_id = get_sub_group_local_id();
  const int g_row = group_id * get_num_sub_groups() + warp_id;
  const int warp_stride = warp_size << 2;

  if (g_row >= M) {
    return;
  }

  local float smem_x[MAX_N];

  for (int i = lane_id; i < N; i += warp_size) {
    smem_x[i] = x[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float sum = 0.0f;
  for (int i = lane_id; i < N; i += warp_stride) {
    float4 vec_a = (float4)(A[g_row * N + i], A[g_row * N + i + 64], A[g_row * N + i + 128], A[g_row * N + i + 192]);
    float4 vec_x = (float4)(smem_x[i], smem_x[i + 64], smem_x[i + 128], smem_x[i + 192]);
    sum += (vec_a.x * vec_x.x + vec_a.y * vec_x.y + vec_a.z * vec_x.z + vec_a.w * vec_x.w);
  }
  sum = sub_group_reduce_add(sum);

  if (lane_id == 0) {
    y[g_row] = alpha * sum + beta * y[g_row];
  }
}