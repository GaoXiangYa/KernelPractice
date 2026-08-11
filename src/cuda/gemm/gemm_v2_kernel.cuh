#include "gemm.cuh"

#define sa(i, j) sa[(i) * BK + j]
#define sb(i, j) sb[(i) * BN + j]

#define MICRO_SIZE 4
#define MICRO_BIT 2
// C[M, N] = A[M, K] * B[K, N]
// BM, BN: block tile size (threads per block in x, y)
// BK: K-dimension tile size
// Each thread computes one output element using shared memory tiling.
// 4x1 marco kernel
// C(i + 0, j) = A(i + 0, k) * B(k, j)
// C(i + 1, j) = A(i + 1, k) * B(k, j)
// C(i + 2, j) = A(i + 2, k) * B(k, j)
// C(i + 3, j) = A(i + 3, k) * B(k, j)

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v2_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K,
                               int lda, int ldb, int ldc) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row_base = blockIdx.y * blockDim.y * MICRO_SIZE;
  const int c_col = blockIdx.x * blockDim.x + tx;
  const int b_col = c_col;

  const int ty0 = ty << MICRO_BIT;
  const int ty1 = ty0 + 1;
  const int ty2 = ty0 + 2;
  const int ty3 = ty0 + 3;

  __shared__ float sa[BM * BK];
  __shared__ float sb[BK * BN];
  float acc[MICRO_SIZE] = {0.0f};
  const int sa_row[TM] = {ty0, ty1, ty2, ty3};
  const int a_row[TM] = {row_base + sa_row[0], row_base + sa_row[1],
                         row_base + sa_row[2], row_base + sa_row[3]};
  const int c_row[TM] = {a_row[0], a_row[1], a_row[2], a_row[3]};
  bool a_pred[TM] = {a_row[0] < M, a_row[1] < M, a_row[2] < M, a_row[3] < M};
  const bool b_pred0 = b_col < N;

  for (int k = 0; k < K; k += BK) {
    int a_col = k + tx;
    bool a_pred_col = a_col < K;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      sa(sa_row[i], tx) =
          a_pred_col && a_pred[i] ? GEMM_A(a_row[i], a_col) : 0.0f;
    }

    int b_row = k + ty;
    bool b_pred = b_row < K;
    sb(ty, tx) = b_pred && b_pred0 ? GEMM_B(b_row, b_col) : 0.0f;
    __syncthreads();
    for (int ik = 0; ik < BK; ++ik) {
      float val_b = sb(ik, tx);
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        acc[i] += sa(sa_row[i], ik) * val_b;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    if (a_pred[i] && b_pred0) {
      GEMM_C(c_row[i], c_col) = acc[i];
    }
  }
}
#undef sa
#undef sb
#undef MICRO_SIZE
#undef MICRO_BIT
