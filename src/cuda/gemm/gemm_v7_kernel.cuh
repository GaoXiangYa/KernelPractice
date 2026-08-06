#define sa(i, j) sa[(i) * SA_STRIDE + j]
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
// 8x4 marco kernel

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v7_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K,
                               int lda, int ldb, int ldc) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row_base = blockIdx.y * blockDim.y * TM;
  const int col_base = blockIdx.x * blockDim.x * TN;
  const int blockSizeX = blockDim.x;
  const int blockSizeY = blockDim.y;

  constexpr int SA_STRIDE = BM + 4;
  __shared__ float sa[BK * SA_STRIDE];
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
    sb_col[i] = tx + i * blockSizeX;
    b_col[i] = col_base + sb_col[i];
    b_pred[i] = b_col[i] < N;
  }

  float acc[TM * TN] = {};

  for (int k = 0; k < K; k += BK) {
    int a_col = k + tx;
    bool a_pred_col = a_col < K;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      sa(tx, sa_row[i]) = a_pred_col && a_pred[i] ? A(a_row[i], a_col) : 0.0f;
    }

#pragma unroll
    for (int off = 0; off < BK; off += blockSizeY) {
      int brow = ty + off;
      int gr = k + brow;
#pragma unroll
      for (int c = 0; c < TN; ++c) {
        sb(brow, sb_col[c]) = (gr < K && b_pred[c]) ? B(gr, b_col[c]) : 0.0f;
      }
    }
    __syncthreads();

    for (int ik = 0; ik < BK; ++ik) {
      float av[TM];
      const float* p = &sa(ik, ty * TM);
      *reinterpret_cast<float4*>(av) = *reinterpret_cast<const float4*>(p);
      *reinterpret_cast<float4*>(av + 4) =
          *reinterpret_cast<const float4*>(p + 4);
      float bv[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        bv[j] = sb(ik, tx * TN + j);
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i * TN + j] += av[i] * bv[j];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    if (a_pred[i] && (col_base + tx * TN + TN - 1 < N)) {
      float4 out = make_float4(acc[i * TN + 0], acc[i * TN + 1],
                               acc[i * TN + 2], acc[i * TN + 3]);
      *reinterpret_cast<float4*>(
          &C[(row_base + sa_row[i]) * ldc + col_base + tx * TN]) = out;
    }
  }
}