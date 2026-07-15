#define GEMM_A(i, j) A[(i) * lda + (j)]
#define GEMM_B(i, j) B[(i) * ldb + (j)]
#define GEMM_C(i, j) C[(i) * ldc + (j)]

#define BM 64
#define BN 64
#define BK 16
#define BM_PAD (BM + 1)
#define BN_PAD (BN + 1)
#define BK_PAD (BK + 1)

#define MICRO_SIZE 4

#define sa(i, j) sa[(i) * BM_PAD + j]
#define sb(i, j) sb[(i) * (BN_PAD) + j]
#define sum(i, j) sum[i * 4 + j]

#define vload(v1,addr)\
    v1 = *((float4 *)(addr));
#define vstore(addr,v1)\
    *((float4 *)(addr)) = v1;

#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;

// C[M, N] = A[M, K] * B[K, N], 16x16x16 tiling
// 1. packend matrix A , matrix B into a 16x16 block
// 2. reduce 2-way bank conflicts
//    subgroup size = 64 has 2 way bank conflicts
//    ly=0, lx=0..15 → addr  0..15  (banks  0..15)
//    ly=1, lx=0..15 → addr 16..31  (banks 16..31)
//    ly=2, lx=0..15 → addr 32..47  (banks  0..15)  ← ly=0 same bank，different address！
//    ly=3, lx=0..15 → addr 48..63  (banks 16..31)  ← ly=1 same bank，different address！
// 3. more workloads per thread. eache thread calculate 4x1 micro kernel
// 4. more workloads per thread. each thread calculate 4x4 sub matrix
// 5. transpose matrix a in shared memory, vload4 load and store,
// 6. warp level parallelism

__kernel void gemm_v9_kernel(__global const float* A, __global const float* B,
                             __global float* C, const int M, const int N,
                             const int K, float alpha, float beta) {
  const int lda = K, ldb = N, ldc = N;

  const int gp_x = get_group_id(0);
  const int gp_y = get_group_id(1);

  // 64x64 tile to divide matrix C
  C = &GEMM_C((gp_y << 6), (gp_x << 6));
  A = &GEMM_A((gp_y << 6), 0);
  B = &GEMM_B(0, (gp_x << 6));

  const int lid = get_local_id(0);
  const int warp_id = lid >> 6;
  const int lane_id = lid & 63;

  // local size 256, warp size 64
  const int warp_row = warp_id & 3;
  const int warp_col = warp_id >> 2;

  // thread in warp layout
  //      col0 col1 col2 ... col15
  // row0
  // row1
  // row2
  // row3
  const int lane_row = lane_id & 3;
  const int lane_col = lane_id >> 2;
  // each thread calculate 4x4 sub matrix, global C index
  const int c_row = (warp_row << 4) + (lane_row << 2);
  const int c_col = (lane_col << 2);
  const int a_row = c_row;
  const int b_col = c_col;

  __local float sa[BK_PAD * BM];
  __local float sb[BK * BN_PAD];

  float4 vec_a;
  float4 vec_b;
  float4 reg_c[MICRO_SIZE];
  reg_c[0] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  reg_c[1] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  reg_c[2] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  reg_c[3] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

  const int b_k = lid >> 4;
  const int b_col_ld = (lid & 15) << 2;

  for (int k = 0; k < K; k += BK) {
    const int a_col = k + lane_col;
    vec_a.x = GEMM_A(a_row, a_col);
    vec_a.y = GEMM_A(a_row + 1, a_col);
    vec_a.z = GEMM_A(a_row + 2, a_col);
    vec_a.w = GEMM_A(a_row + 3, a_col);
    *(float4*)&sa(lane_col, a_row) = vec_a;

    vload(vec_b, &GEMM_B(k + b_k, b_col_ld));
    vstore(&sb(b_k, b_col_ld), vec_b);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int ik = 0; ik < BK; ++ ik) {
      vload(vec_b, &sb(ik, b_col));
      vload(vec_a, &sa(ik, a_row));

      vscal(reg_c[0], vec_a, vec_b.x);
      vscal(reg_c[1], vec_a, vec_b.y);
      vscal(reg_c[2], vec_a, vec_b.z);
      vscal(reg_c[3], vec_a, vec_b.w);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float4 vec_c;
  vload(vec_c, &GEMM_C(c_row, c_col));
  vec_c.x = reg_c[0].x * alpha + vec_c.x * beta;
  vec_c.y = reg_c[1].x * alpha + vec_c.y * beta;
  vec_c.z = reg_c[2].x * alpha + vec_c.z * beta;
  vec_c.w = reg_c[3].x * alpha + vec_c.w * beta;
  vstore(&GEMM_C(c_row, c_col), vec_c);

  vload(vec_c, &GEMM_C(c_row + 1, c_col));
  vec_c.x = reg_c[0].y * alpha + vec_c.x * beta;
  vec_c.y = reg_c[1].y * alpha + vec_c.y * beta;
  vec_c.z = reg_c[2].y * alpha + vec_c.z * beta;
  vec_c.w = reg_c[3].y * alpha + vec_c.w * beta;
  vstore(&GEMM_C(c_row + 1, c_col), vec_c);
  
  vload(vec_c, &GEMM_C(c_row + 2, c_col));
  vec_c.x = reg_c[0].z * alpha + vec_c.x * beta;
  vec_c.y = reg_c[1].z * alpha + vec_c.y * beta;
  vec_c.z = reg_c[2].z * alpha + vec_c.z * beta;
  vec_c.w = reg_c[3].z * alpha + vec_c.w * beta;
  vstore(&GEMM_C(c_row + 2, c_col), vec_c);
  
  vload(vec_c, &GEMM_C(c_row + 3, c_col));
  vec_c.x = reg_c[0].w * alpha + vec_c.x * beta;
  vec_c.y = reg_c[1].w * alpha + vec_c.y * beta;
  vec_c.z = reg_c[2].w * alpha + vec_c.z * beta;
  vec_c.w = reg_c[3].w * alpha + vec_c.w * beta;
  vstore(&GEMM_C(c_row + 3, c_col), vec_c);
}