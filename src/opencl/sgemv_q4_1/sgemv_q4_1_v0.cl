// y[M] = alpha * dequant(A_q4_1)[M, N] * x[N] + beta * y[M]
// Q4_1 block (block_k floats): [d: float][m: float][qs: block_k/2 bytes]
// dequant: even col -> low nibble, odd col -> high nibble.
// Naive v0: one work-item per row, serial over blocks and nibbles.

#define BLOCK_BYTES(block_k) ((sizeof(float) << 1) + ((block_k) >> 1))

__kernel void sgemv_q4_1_v0_kernel(
    __global const uchar* A, __global const float* x, __global float* y,
    const int M, const int N, const int block_k, float alpha, float beta) {
  const int row = get_global_id(0);
  if (row >= M) {
    return;
  }

  const int blocks_per_row = N / block_k;
  const int block_bytes = BLOCK_BYTES(block_k);
  const __global uchar* row_a = A + (size_t) row * blocks_per_row * block_bytes;

  float sum = 0.0f;
  for (int b = 0; b < blocks_per_row; ++b) {
    const __global uchar* blk = row_a + b * block_bytes;
    const float d = *(__global const float*) blk;
    const float m = *(__global const float*) (blk + sizeof(float));
    const __global uchar* qs = blk + (sizeof(float) << 1);
    const int col = b * block_k;

    for (int i = 0; i < block_k / 2; ++i) {
      const uchar packed = qs[i];
      const float v0 = (float) (packed & 0x0F);
      const float v1 = (float) (packed >> 4);
      sum += (v0 * d + m) * x[col + 2 * i];
      sum += (v1 * d + m) * x[col + 2 * i + 1];
    }
  }

  y[row] = alpha * sum + beta * y[row];
}
