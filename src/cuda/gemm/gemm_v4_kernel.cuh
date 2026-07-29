#define sa(i, j) sa[(i) * BK + j]
#define sb(i, j) sb[(i) * BN + j]
#define acc(i, j) acc[(i) * TN + j]
#define A(i, j) A[(i) * lda + j]
#define B(i, j) B[(i) * ldb + j]
#define C(i, j) C[(i) * ldc + j]

// #define vload(vec, mat, pred)
// C[M, N] = A[M, K] * B[K, N]
// BM, BN: block tile size (threads per block in x, y)
// BK: K-dimension tile size
// Each thread computes one output element using shared memory tiling.
// 4x4 marco kernel

#define vstore(addr, vec) *((float4*) addr) = vec
#define vload(vec, addr) vec = *((float4*) (addr))

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v4_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K,
                               int lda, int ldb, int ldc) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row_base = blockIdx.y * blockDim.y * TM;
  const int col_base = blockIdx.x * blockDim.x * TN;

  __shared__ float sa[BM * BK];
  __shared__ float sb[BK * BN];

  int sa_row[TM];
  int a_row[TM];
  bool a_pred[TM];
  const int sa_row_base = ty * TM;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    sa_row[i] = sa_row_base + i;
    a_row[i] = row_base + sa_row[i];
    a_pred[i] = a_row[i] < M;
  }

  int sb_col[TN];
  int b_col[TN];
  bool b_pred[TN];
  const int sb_col_base = tx * TN;
#pragma unroll
  for (int i = 0; i < TN; ++i) {
    sb_col[i] = sb_col_base + i;
    b_col[i] = col_base + sb_col[i];
    b_pred[i] = b_col[i] < N;
  }

  float acc[TM * TN];
  memset(acc, 0, sizeof(float) * TM * TN);

  for (int k = 0; k < K; k += BK) {
    int a_col = k + tx;
    bool a_pred_col = a_col < K;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      sa(sa_row[i], tx) = a_pred_col && a_pred[i] ? A(a_row[i], a_col) : 0.0f;
    }

    int b_row = k + ty;
    bool b_pred_row = b_row < K;
#pragma unroll
    for (int i = 0; i < TN; ++i) {
      // reduce bank conflicts!
      sb(ty, i * blockDim.x + tx) =
          b_pred_row && b_pred[i] ? B(b_row, b_col[i]) : 0.0f;
    }

    __syncthreads();

    for (int ik = 0; ik < BK; ++ik) {
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        float a_val = sa(sa_row[i], ik);
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          float b_val = sb(ik, j * blockDim.x + tx);
          acc(i, j) += a_val * b_val;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      if (a_pred[i] && b_pred[j]) {
        C(a_row[i], b_col[j]) = acc(i, j);
      }
    }
  }
}