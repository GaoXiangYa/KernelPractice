#define GEMM_A(i, j) A[(i) * lda + (j)]
#define GEMM_B(i, j) B[(i) * ldb + (j)]
#define GEMM_C(i, j) C[(i) * ldc + (j)]

#define BM 16
#define BN 16
#define BK 16
#define BK_PAD (BK + 1)

#define sa(i, j) sa[(i) * BK_PAD + j]
#define sb(i, j) sb[(i) * BN + j]

// C[M, N] = A[M, K] * B[K, N], 16x16x16 tiling
// 1. packend matrix A , matrix B into a 16x16 block
// 2. reduce 2-way bank conflicts
//  subgroup size = 64 has 2 way bank conflicts
//  ly=0, lx=0..15 → addr  0..15  (banks  0..15)
//  ly=1, lx=0..15 → addr 16..31  (banks 16..31)
//  ly=2, lx=0..15 → addr 32..47  (banks  0..15)  ← ly=0 same bank，different address！
//  ly=3, lx=0..15 → addr 48..63  (banks 16..31)  ← ly=1 same bank，different address！
__kernel void gemm_v2_kernel(__global const float* A, __global const float* B,
                             __global float* C, const int M, const int N,
                             const int K, float alpha, float beta) {
  const int lda = K, ldb = N, ldc = N;
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int gx = get_group_id(0) * get_local_size(0) + lx;
  const int gy = get_group_id(1) * get_local_size(1) + ly;

  __local float sa[BM * BK_PAD];
  __local float sb[BK_PAD * BN];

  float sum = 0.0f;
  for (int k = 0; k < K; k += BK) {
    const int base = k;

    const int ga_x = base +lx;
    if (gy < M && ga_x < K) {
      sa(ly, lx) = GEMM_A(gy, ga_x);
    } else {
      sa(ly, lx) = 0.0f;
    }
    const int gb_y = base +ly;
    if (gb_y < K && gx < N) {
      sb(ly, lx) = GEMM_B(gb_y, gx);
    } else {
      sb(ly, lx) = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int ik = 0; ik < BK; ++ ik) {
      sum += sa(ly, ik) * sb(ik, lx);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }


  if (gy < M && gx < N) {
    GEMM_C(gy, gx) = alpha * sum + beta * GEMM_C(gy, gx);
  }
}