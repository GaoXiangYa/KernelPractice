#include "gemm.cuh"

#define sa(i, j) sa[(i) * BM + j]
#define sb(i, j) sb[(i) * BN + j]

#define MICRO_SIZE 4
#define MICRO_BIT 2

// #define vload(vec, mat, pred)
// C[M, N] = A[M, K] * B[K, N]
// BM, BN: block tile size (threads per block in x, y)
// BK: K-dimension tile size
// Each thread computes one output element using shared memory tiling.
// 4x1 marco kernel
// C(i + 0, j) = A(i + 0, k) * B(k, j)
// C(i + 1, j) = A(i + 1, k) * B(k, j)
// C(i + 2, j) = A(i + 2, k) * B(k, j)
// C(i + 3, j) = A(i + 3, k) * B(k, j)

#define vstore(addr, vec) *((float4*) addr) = vec
#define vload(vec, addr) vec = *((float4*) (addr))

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v3_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K,
                               int lda, int ldb, int ldc) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row_base = blockIdx.y * blockDim.y * MICRO_SIZE;
  const int c_col = blockIdx.x * blockDim.x + tx;
  const int b_col = c_col;

  const int ty0 = ty << MICRO_BIT;
  const int ty1 = ty + 1;
  const int ty2 = ty + 2;
  const int ty3 = ty + 3;

  __shared__ float sa[BK * BM];
  __shared__ float sb[BK * BN];
  float4 acc = (float4){0.0f, 0.0f, 0.0f, 0.0f};
  const int4 a_row =
      (int4){row_base + ty0, row_base + ty1, row_base + ty2, row_base + ty3};
  const int4 c_row = (int4){a_row.x, a_row.y, a_row.z, a_row.w};
  float4 a_vec;
  bool a_pred[TM] = {a_row.x < M, a_row.y < M, a_row.z < M, a_row.w < M};
  const bool b_pred0 = b_col < N;

  for (int k = 0; k < K; k += BK) {
    int a_col = k + tx;
    bool a_pred_col = a_col < K;
    a_vec.x = a_pred_col && a_pred[0] ? GEMM_A(a_row.x, a_col) : 0.0f;
    a_vec.y = a_pred_col && a_pred[1] ? GEMM_A(a_row.y, a_col) : 0.0f;
    a_vec.z = a_pred_col && a_pred[2] ? GEMM_A(a_row.z, a_col) : 0.0f;
    a_vec.w = a_pred_col && a_pred[3] ? GEMM_A(a_row.w, a_col) : 0.0f;
    vstore(&sa(tx, ty0), a_vec);

    int b_row = k + ty;
    bool b_pred = b_row < K;
    sb(ty, tx) = b_pred && b_pred0 ? GEMM_B(b_row, b_col) : 0.0f;
    __syncthreads();
    for (int ik = 0; ik < BK; ++ik) {
      float val_b = sb(ik, tx);
      vload(a_vec, &sa(ik, ty0));
      acc.x += a_vec.x * val_b;
      acc.y += a_vec.y * val_b;
      acc.z += a_vec.z * val_b;
      acc.w += a_vec.w * val_b;
    }
    __syncthreads();
  }
  vstore((&GEMM_C(c_row.x, tx)), acc);
}