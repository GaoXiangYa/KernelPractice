#define BLOCK_K 32
#define BM 16
#define BN 16
#define BK BLOCK_K

#define B(i, j) B[(i) * N + j]
#define C(i, j) C[(i) * N + j]

#define sa(i, j) sa[(i) * BK + j]
#define sb(i, j) sb[(i) * BN + j]

inline void deq_q4_1_block(__global const uchar* qs, float d, float m,
                           __local float* dest) {
    for (int i = 0; i < BLOCK_K / 2; ++i) {
        uchar packed = qs[i];
        dest[2 * i + 0] = (float)(packed & 0x0F) * d + m;
        dest[2 * i + 1] = (float)(packed >> 4)   * d + m;
    }
}

__kernel void gemm_q4_1_v0_kernel(
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
    __local float sa[BM * BK];
    __local float sb[BK * BN];
    float acc = 0.0f;

    for (int k = 0; k < K; k += BK) {
        if (gy < M) {
            const int A_block_col = k >> 5;
            __global uchar* A_block_per_row = (__global uchar*)A + row_offset;
            __global uchar* tile_a_block = (__global uchar*)A_block_per_row + A_block_col * block_size;
            float d = *((__global float*)tile_a_block);
            float m = *((__global float*)(tile_a_block + sizeof(float)));
            __global uchar* q = (__global uchar*)tile_a_block + sizeof(float) + sizeof(float);
            deq_q4_1_block(q, d, m, sa + ly * BK);
        }

        #pragma unroll
        for (int off = 0; off < BLOCK_K; off += 16) {
            const int b_row = k + off + ly;
            if (gx < N && b_row < K) {
                sb(ly + off, lx) = B(b_row, gx);
            } else {
                sb(ly + off, lx) = 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int ik = 0; ik < BK; ++ ik) {
            acc += sa(ly, ik) * sb(ik, lx);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gy < M && gx < N) {
        C(gy, gx) = acc * alpha + beta * C(gy, gx);
    }
}
