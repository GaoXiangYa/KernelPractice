#define BLOCK_K 32
#define BLOCK_K_HALF 16
#define BM 32
#define BN 8
#define BK BLOCK_K

#define B(i, j) B[(i) * N + j]
#define C(i, j) C[(i) * N + j]

#define sa(i, j) sa[(i) * BLOCK_K_HALF + j]
#define sb(i, j) sb[(i) * BN + j]

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

__kernel void gemm_q4_1_v3_kernel(
    __global const uchar* A,
    __global const float* B,
    __global       float* C,
    const int M, const int N, const int K,
    const int block_k,                  // BLOCK_K (quantization group size)
    const float alpha, const float beta
) {
    // tile: 16x32
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gy = get_group_id(1) * get_local_size(1) + ly;
    const int gx = get_group_id(0) * get_local_size(0) + lx;
    const int block_size = (sizeof(float) << 1) + (BLOCK_K >> 1);
    int row_stride = (K >> 5) * block_size;
    int row_offset = gy * row_stride;
    __local uchar sa[BM * BLOCK_K_HALF];
    __local float sb[BK * BN];
    float d_frag;
    float m_frag;
    float acc = 0.0f;

    for (int k = 0; k < K; k += BK) {
        if (gy < M) {
            const int A_block_col = k >> 5;
            __global uchar* A_block_per_row = (__global uchar*)A + row_offset;
            __global uchar* tile_a_block = (__global uchar*)A_block_per_row + A_block_col * block_size;
            float d = *((__global float*)tile_a_block);
            float m = *((__global float*)(tile_a_block + sizeof(float)));
            d_frag = d;
            m_frag = m;
            __global uchar* q = (__global uchar*)tile_a_block + sizeof(float) + sizeof(float);
            load_tiled(q, sa + ly * BLOCK_K_HALF);
        }

        const int b_row = k + ly;
        if (gx < N && b_row < K) {
            sb(ly, lx) = B(b_row, gx);
        } else {
            sb(ly, lx) = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (int ik = 0; ik < BK; ++ ik) {
            uchar packed = sa(ly, (ik >> 1));
            float q = (float)((ik & 1) ? (packed >> 4) : (packed & 0x0F));
            float val_b = sb(ik, lx);
            sum0 += q * val_b;
            sum1 += val_b;
        }
        acc += (d_frag * sum0 + m_frag * sum1);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gy < M && gx < N) {
        C(gy, gx) = acc * alpha + beta * C(gy, gx);
    }
}
