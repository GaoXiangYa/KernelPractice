#define BLOCK_K 32
#define BLOCK_K_HALF 16
#define BM 128
#define BN 32
#define BK BLOCK_K

#define TM 4 
#define TN 4

#define B(i, j) B[(i) * N + j]
#define C(i, j) C[(i) * N + j]

#define sa(i, j) sa[(i) * BLOCK_K_HALF + j]
#define sb(i, j) sb[(i) * BN + j]
#define acc(i, j) acc[(i) * TN + j]

inline void deq_q4_1_block(__global const uchar* qs, __local float* dest) {
    for (int i = 0; i < BLOCK_K / 2; ++i) {
        uchar packed = qs[i];
        dest[2 * i + 0] = (float)(packed & 0x0F);
        dest[2 * i + 1] = (float)(packed >> 4);
    }
}

inline void load_tiled(__global const uchar* qs, __local uchar* dest) {
    for (int i = 0; i < BLOCK_K_HALF; ++ i) {
        dest[i] = qs[i];
    }
}

// each thread calculate 4x4 micro kernel
__kernel void gemm_q4_1_v2_kernel(
    __global const uchar* A,
    __global const float* B,
    __global       float* C,
    const int M, const int N, const int K,
    const int block_k,                  // BLOCK_K (quantization group size)
    const float alpha, const float beta
) {
    // tile: 32 * 128, tx: 0~8, ty: 0~32, tm: 4, tn: 4
    const int lx_base = get_local_id(0) << 2;
    const int lx[TN] = {lx_base + 0, lx_base + 1, lx_base + 2, lx_base + 3};

    const int ly_base = get_local_id(1) << 2;
    const int ly[TM] = {ly_base + 0, ly_base + 1, ly_base + 2, ly_base + 3};

    const int lsz_y = get_local_size(1) * TM;
    const int lsz_x = get_local_size(0) * TN;

    const int group_y = get_group_id(1);
    const int group_x = get_group_id(0);

    const int gy[4] = {group_y * lsz_y + ly[0], group_y * lsz_y + ly[1], 
                        group_y * lsz_y + ly[2], group_y * lsz_y + ly[3]};
    const bool gy_pred[4] = {gy[0] < M, gy[1] < M, gy[2] < M, gy[3] < M};
    const int gx[4] = {group_x * lsz_x + lx[0], group_x * lsz_x + lx[1],
                        group_x * lsz_x + lx[2], group_x * lsz_x + lx[3]};
    const bool gx_pred[4] = {gx[0] < N, gx[1] < N, gx[2] < N, gx[3] < N};
    
    const int block_size = (sizeof(float) << 1) + (BLOCK_K >> 1);
    int row_stride = (K >> 5) * block_size;
    int row_offset[4] = {gy[0] * row_stride, gy[1] * row_stride, gy[2] * row_stride, gy[3] * row_stride};
    __local uchar sa[BM * BLOCK_K_HALF];
    __local float sb[BK * BN];
    float d_frag[TM];
    float m_frag[TM];
    float acc[TM * TN] = {0.0f};

    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for (int i = 0; i < TM; ++ i) {
            if (gy_pred[i]) {
                const int A_block_col = k >> 5;
                __global uchar* A_block_per_row = (__global uchar*)A + row_offset[i];
                __global uchar* tile_a_block = (__global uchar*)A_block_per_row + A_block_col * block_size;
                float d = *((__global float*)tile_a_block);
                float m = *((__global float*)(tile_a_block + sizeof(float)));
                d_frag[i] = d;
                m_frag[i] = m;
                __global uchar* q = (__global uchar*)tile_a_block + sizeof(float) + sizeof(float);
                load_tiled(q, sa + ly[i] * BLOCK_K_HALF);
            }
        }

        int b_k   = get_local_id(1);
        int b_row = k + b_k;
        if (b_row < K) {
            for (int j = 0; j < TN; ++j) {
                sb(b_k, lx[j]) = gx_pred[j] ? B(b_row, gx[j]) : 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (int ik = 0; ik < BK; ++ ik) {
            #pragma unroll
            for (int i = 0; i < TM; ++ i) {
                uchar packed = sa(ly[i], (ik >> 1));
                float val_a = (float)((ik & 1) ? (packed >> 4) : (packed & 0x0F));
            #pragma unroll
                for (int j = 0; j < TN; ++ j) {
                    float val_b = sb(ik, lx[j]);
                    acc(i, j) += (d_frag[i] * val_a * val_b + m_frag[i] * val_b);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int i = 0; i < TM; ++ i) {
    #pragma unroll
        for (int j = 0; j < TN; ++ j) {
            if (gy_pred[i] && gx_pred[j]) {
                C(gy[i], gx[j]) = acc(i, j) * alpha + beta * C(gy[i], gx[j]);
            }
        }
    }
}
