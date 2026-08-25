// y[M] = alpha * A[M, N] * x[N] + beta * y[M]
// Naive: one work-item per row, serial dot product over N.
__kernel void sgemv_v0_kernel(__global const float* A, __global const float* x,
                              __global float* y, const int M, const int N,
                              float alpha, float beta) {
  const int row = get_global_id(0);
  if (row >= M) {
    return;
  }

  float sum = 0.0f;
  for (int col = 0; col < N; ++col) {
    sum += A[row * N + col] * x[col];
  }
  y[row] = alpha * sum + beta * y[row];
}
