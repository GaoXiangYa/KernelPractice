#define GEMM_A(i, j) A[(i) * lda + (j)]
#define GEMM_B(i, j) B[(i) * ldb + (j)]
#define GEMM_C(i, j) C[(i) * ldc + (j)]

#define BM 64
#define BN 16
#define BK 16
#define BK_PAD (BK + 1)

#define MICRO_SIZE 4

#define sa(i, j) sa[(i) * BK_PAD + j]
#define sb(i, j) sb[(i) * (BN + 1) + j]

// C[M, N] = A[M, K] * B[K, N], 16x16x16 tiling
// 1. packend matrix A , matrix B into a 16x16 block
// 2. reduce 2-way bank conflicts
//    subgroup size = 64 has 2 way bank conflicts
//    ly=0, lx=0..15 → addr  0..15  (banks  0..15)
//    ly=1, lx=0..15 → addr 16..31  (banks 16..31)
//    ly=2, lx=0..15 → addr 32..47  (banks  0..15)  ← ly=0 same bank，different address！
//    ly=3, lx=0..15 → addr 48..63  (banks 16..31)  ← ly=1 same bank，different address！
// 3. more workloads per thread. 4x1 micro kernel.

inline static void load_matrixA_to_shared(__global const float* A, const int gy, const int gx, __local float* sa, const int ly, const int lx, const int M, const int K) {
  const int lda = K;
  if (gy < M && gx < K) {
    sa(ly, lx) = GEMM_A(gy, gx);
  } else {
    sa(ly, lx) = 0.0f;
  }
}

inline static void load_matrixB_to_shared(__global const float* B, const int gy, const int gx, __local float* sb, const int ly, const int lx, const int K, const int N) {
  const int ldb = N;
  if (gy < K && gx < N) {
    sb(ly, lx) = GEMM_B(gy, gx);
  } else {
    sb(ly, lx) = 0.0f;
  }
}

inline static void load_matrixC(__global float* C, const int gy, const int gx, const int M, const int N, float sum, float alpha, float beta) {
  const int ldc = N;
  if (gy < M && gx < N) {
    GEMM_C(gy, gx) = alpha * sum + beta * GEMM_C(gy, gx);
  }
}

__kernel void gemm_v3_kernel(__global const float* A, __global const float* B,
                             __global float* C, const int M, const int N,
                             const int K, float alpha, float beta) {
  const int lda = K, ldb = N, ldc = N;
  const int lsz1 = get_local_size(1);
  const int gp1 = get_group_id(1);
  const int gp1_size = gp1 * lsz1 * MICRO_SIZE;
  const int ly_base = get_local_id(1);
  const int lx = get_local_id(0);
  const int gx = get_group_id(0) * get_local_size(0) + lx;

  const int ly0 = ly_base << 2;
  const int ly1 = ly0 + 1;
  const int ly2 = ly0 + 2;
  const int ly3 = ly0 + 3;
  
  const int gy0 = gp1_size + ly0;
  const int gy1 = gp1_size + ly1;
  const int gy2 = gp1_size + ly2;
  const int gy3 = gp1_size + ly3;

  __local float sa[BM * BK_PAD];
  __local float sb[BK_PAD * BN];

  float sum[MICRO_SIZE] = {0.0f};

  for (int k = 0; k < K; k += BK) {
    const int base = k;
    
    const int ga_x = base + lx;
    load_matrixA_to_shared(A, gy0, ga_x, sa, ly0, lx, M, K);
    load_matrixA_to_shared(A, gy1, ga_x, sa, ly1, lx, M, K);
    load_matrixA_to_shared(A, gy2, ga_x, sa, ly2, lx, M, K);
    load_matrixA_to_shared(A, gy3, ga_x, sa, ly3, lx, M, K);
    
    load_matrixB_to_shared(B, base + ly_base, gx, sb, ly_base, lx, K, N);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int ik = 0; ik < BK; ++ ik) {
      float val_b = sb(ik, lx);
      sum[0] += sa(ly0, ik) * val_b;
      sum[1] += sa(ly1, ik) * val_b;
      sum[2] += sa(ly2, ik) * val_b;
      sum[3] += sa(ly3, ik) * val_b;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  load_matrixC(C, gy0, gx, M, N, sum[0], alpha, beta);
  load_matrixC(C, gy1, gx, M, N, sum[1], alpha, beta);
  load_matrixC(C, gy2, gx, M, N, sum[2], alpha, beta);
  load_matrixC(C, gy3, gx, M, N, sum[3], alpha, beta);
}