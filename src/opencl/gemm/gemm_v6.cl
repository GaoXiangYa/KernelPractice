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

__kernel void gemm_v6_kernel(__global const float* A, __global const float* B,
                             __global float* C, const int M, const int N,
                             const int K, float alpha, float beta) {
  const int lda = K, ldb = N, ldc = N;

  const int lsz0 = get_local_size(0);
  const int gp0 = get_group_id(0);
  const int gp0_size = gp0 * lsz0 * MICRO_SIZE;
  const int lx_base = get_local_id(0);

  const int lx0 = lx_base << 2;
  const int lx1 = lx0 + 1;
  const int lx2 = lx0 + 2;
  const int lx3 = lx0 + 3;

  const int gx0 = gp0_size + lx0;
  const int gx1 = gp0_size + lx1;
  const int gx2 = gp0_size + lx2;
  const int gx3 = gp0_size + lx3;

  const int lsz1 = get_local_size(1);
  const int gp1 = get_group_id(1);
  const int gp1_size = gp1 * lsz1 * MICRO_SIZE;
  const int ly_base = get_local_id(1);

  const int ly0 = ly_base << 2;
  const int ly1 = ly0 + 1;
  const int ly2 = ly0 + 2;
  const int ly3 = ly0 + 3;
  
  const int gy0 = gp1_size + ly0;
  const int gy1 = gp1_size + ly1;
  const int gy2 = gp1_size + ly2;
  const int gy3 = gp1_size + ly3;

  __local float sa[BM_PAD * BK];
  __local float sb[BK * BN_PAD];

  float4 vec_b;
  float4 vec_a;

  float4 reg_c[MICRO_SIZE];
  reg_c[0] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  reg_c[0] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  reg_c[0] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  reg_c[0] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);


  for (int k = 0; k < K; k += BK) {
    const int base = k;

    // load to shared memory 
    int ga_x = base + lx_base;
    vec_a.x = GEMM_A(gy0, ga_x);
    vec_a.y = GEMM_A(gy1, ga_x);
    vec_a.z = GEMM_A(gy2, ga_x);
    vec_a.w = GEMM_A(gy3, ga_x);
    // transpose tiled a
    *(float4*)&sa(lx_base, ly0) = vec_a;
 
    int gb_y = base + ly_base;
    vload(vec_b, &GEMM_B(gb_y, gx0));
    vstore(&sb(ly_base, lx0), vec_b);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int ik = 0; ik < BK; ++ ik) {
      vload(vec_b, &sb(ik, lx0));
      vload(vec_a, &sa(ik, ly0));

      vscal(reg_c[0], vec_a, vec_b.x);
      vscal(reg_c[1], vec_a, vec_b.y);
      vscal(reg_c[2], vec_a, vec_b.z);
      vscal(reg_c[3], vec_a, vec_b.w);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  float4 vec_c;
  vload(vec_c, &GEMM_C(gy0, gx0));
  vec_c.x = reg_c[0].x * alpha + vec_c.x * beta;
  vec_c.y = reg_c[1].x * alpha + vec_c.y * beta;
  vec_c.z = reg_c[2].x * alpha + vec_c.z * beta;
  vec_c.w = reg_c[3].x * alpha + vec_c.w * beta;
  vstore(&GEMM_C(gy0, gx0), vec_c);

  vload(vec_c, &GEMM_C(gy1, gx0));
  vec_c.x = reg_c[0].y * alpha + vec_c.x * beta;
  vec_c.y = reg_c[1].y * alpha + vec_c.y * beta;
  vec_c.z = reg_c[2].y * alpha + vec_c.z * beta;
  vec_c.w = reg_c[3].y * alpha + vec_c.w * beta;
  vstore(&GEMM_C(gy1, gx0), vec_c);
  
  vload(vec_c, &GEMM_C(gy2, gx0));
  vec_c.x = reg_c[0].z * alpha + vec_c.x * beta;
  vec_c.y = reg_c[1].z * alpha + vec_c.y * beta;
  vec_c.z = reg_c[2].z * alpha + vec_c.z * beta;
  vec_c.w = reg_c[3].z * alpha + vec_c.w * beta;
  vstore(&GEMM_C(gy2, gx0), vec_c);
  
  vload(vec_c, &GEMM_C(gy3, gx0));
  vec_c.x = reg_c[0].w * alpha + vec_c.x * beta;
  vec_c.y = reg_c[1].w * alpha + vec_c.y * beta;
  vec_c.z = reg_c[2].w * alpha + vec_c.z * beta;
  vec_c.w = reg_c[3].w * alpha + vec_c.w * beta;
  vstore(&GEMM_C(gy3, gx0), vec_c);
}