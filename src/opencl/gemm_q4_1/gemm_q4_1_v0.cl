// ===========================================================================
// GEMM with Q4_1 quantized matrix A:  C[M,N] = dequant(A) × B[K,N]
//
// Q4_1 format (per block of BLOCK_K elements in each row of A):
//   d[float]  — scale factor
//   m[float]  — bias / minimum
//   qs[BLOCK_K/2 bytes] — quantized values, 2 int4 per byte (low nibble first)
//
// Dequant:  val = (qs[i] & 0x0F) * d + m      (low)
//           val = (qs[i] >> 4)   * d + m      (high)
//
// A stored row-major: all blocks for row 0, then all blocks for row 1, ...
// Per row:  (sizeof(float)*2 + BLOCK_K/2) * ceil(K / BLOCK_K)  bytes
// ===========================================================================

// Number of elements quantized together  (typically 32)
#define BLOCK_K 32
#define BM 16
#define BN 16
#define BK BLOCK_K

#define B(i, j) B[(i) * BN + j]
#define C(i, j) C[(i) * BN + j]

#define sa(i, j) sa[(i) * BK + j]
#define sb(i, j) sb[(i) * BN + j]

struct q4_1_block{
    float d;
    float m;
    uint8 d[16];
}

// ---- Dequantize one block of A into a register tile ----
// A_row points to the start of a q4_1 block.
// dest must have at least BLOCK_K float slots.
inline void deq_q4_1_block(__global const uchar* qs, float d, float m,
                           __local float* dest) {
    for (int i = 0; i < BLOCK_K / 2; ++i) {
        uchar packed = qs[i];
        dest[2 * i + 0] = (float)(packed & 0x0F) * d + m;
        dest[2 * i + 1] = (float)(packed >> 4)   * d + m;
    }
}

// ---- Kernel signature ----
// A   : quantized matrix A in Q4_1 layout  (M rows × ceil(K/BLOCK_K) blocks)
// B   : regular float matrix B  [K, N]
// C   : output  [M, N]
// offset_A : byte offset to first element in A buffer (for non-contiguous)
// offset_B, offset_C : element offset for B, C
__kernel void gemm_q4_v0_kernel(
    __global const uchar* A,
    __global const float* B,
    __global       float* C,
    const int M, const int N, const int K,
    const int block_k,                  // BLOCK_K (quantization group size)
    const float alpha, const float beta
) {
    // ---- your tiled GEMM here ----
    //
    // Steps per work-group:
    //   1. determine tile position (row/col in C)
    //   2. for each k-tile:
    //      a. dequantize A[rows][k_tile] into registers / local memory
    //      b. load B[k_tile][cols] from global
    //      c. compute partial sum += deq_A × B
    //   3. write C[tile_rows][tile_cols] = alpha * sum + beta * C_prev
    //
    // Helper to read one int4 value from a packed nibble:
    //   (qs[byte_idx] >> nibble_select) & 0x0F    nibble_select = 0 (low) or 4 (high)
    //
    // To locate block header for row r, k-slice s:
    //   blocks_per_row = K / block_k  (assumes K % block_k == 0)
    //   block_stride   = sizeof(float)*2 + block_k/2   (scale + bias + packed qs)
    //   A_block = A + r * blocks_per_row * block_stride  +  s * block_stride
    //   block_d = *(float*)(A_block)
    //   block_m = *(float*)(A_block + sizeof(float))
    //   block_q = A_block + sizeof(float)*2

    // TODO: your code here
    // tile: 16x32
    __global struct q4_1_block* A_block = (__global struct q4_1_block*)A;
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gy = get_group_id(1) * get_local_size(1) + ly;
    const int gx = get_group_id(0) * get_local_size(0) + lx;
    const int block_stride = sizeof(float) * 2 + BLOCK_K / 2;
    __local float sa[BM * BK];
    __local float sb[BK * BN];
    float acc = 0.0f;

    for (int k = 0; k < K; k += BK) {
        if (gy < M) {
            const int A_block_col = k >> 5;
            __global struct q4_1_block* A_block_per_row = (__global struct q4_1_block*)((__global char*)A_block + gy * K * block_stride);
            __global struct q4_1_block* tile_a_block = (__global struct q4_1_block*)((__global char*)A_block_per_row + A_block_col * block_stride);
            // __private float sa[BK];
            float d = tile_a_block.d;
            float m = tile_a_block.m;
            __global uchar* q = (__global uchar*)((__global char*)tile_a_block + sizeof(float) + sizeof(float));
            deq_q4_1_block(q, d, m, sa + ly * BK * sizeof(float));
        }

        const int b_row = k + ly;
        if (gx < N && b_row < K) {
            sb[ly * BN + lx] = B(b_row, gx);
        } else {
            sb[ly * BN + lx] = 0.0f;
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
