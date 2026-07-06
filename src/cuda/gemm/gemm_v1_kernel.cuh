#include "gemm.cuh"

// C[M, N] = A[M, K] * B[K, N]
// BM, BN: block tile size (threads per block in x, y)
// BK: K-dimension tile size
// Each thread computes one output element using shared memory tiling.
template <int BM, int BN, int BK>
__global__ void gemm_v1_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K,
                               int lda, int ldb, int ldc) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  float sum = 0.0f;

  for (int base = 0; base < K; base += BK) {
    // cooperative load A tile (BM x BK): only tx < BK participate
    if (tx < BK) {
      int k_idx = base + tx;
      if (row < M && k_idx < K)
        As[ty * BK + tx] = GEMM_A(row, k_idx);
      else
        As[ty * BK + tx] = 0.0f;
    }

    // cooperative load B tile (BK x BN): only ty < BK participate
    if (ty < BK) {
      int k_idx = base + ty;
      if (k_idx < K && col < N)
        Bs[ty * BN + tx] = GEMM_B(k_idx, col);
      else
        Bs[ty * BN + tx] = 0.0f;
    }

    __syncthreads();

    // compute dot product from shared memory
    for (int k = 0; k < BK; ++k) {
      sum += As[ty * BK + k] * Bs[k * BN + tx];
    }

    __syncthreads();
  }

  if (col < N && row < M) {
    GEMM_C(row, col) = sum;
  }
}
