#include "test_utils.h"
#include "gemm_q4_1.h"
#include <cmath>
#include <cstring>
#include <vector>

// ===========================================================================
// Q4_1 packing
// ===========================================================================
static void pack_q4_1(const float* src, int k, int block_k,
                      unsigned char* dst) {
    float d = 0.0f, m = 0.0f;
    for (int i = 0; i < k; ++i) {
        float v = src[i];
        m = fminf(m, v);
        d = fmaxf(d, v - m);
    }
    if (d == 0.0f) d = 1.0f;
    memcpy(dst, &d, sizeof(float)); dst += sizeof(float);
    memcpy(dst, &m, sizeof(float)); dst += sizeof(float);
    for (int i = 0; i < k; i += 2) {
        int q0 = (int)roundf((src[i]     - m) / d * 15.0f);
        int q1 = (i+1<k) ? (int)roundf((src[i+1] - m) / d * 15.0f) : 0;
        q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
        q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
        *dst++ = (unsigned char)(q0 | (q1 << 4));
    }
}

// ===========================================================================
// Dequantize helper (CPU)
// ===========================================================================
static float deq_nibble(const unsigned char* qs, int idx, float d, float m) {
    unsigned char b = qs[idx / 2];
    int nib = (idx & 1) ? (b >> 4) : (b & 0x0F);
    return (float)nib * d + m;
}

// ===========================================================================
// CPU reference: full GEMM with Q4_1 dequantization
// ===========================================================================
static void gemm_q4_ref(const unsigned char* A_q4, const float* B, float* C,
                        int M, int N, int K, int block_k,
                        float alpha, float beta) {
    int blocks_per_row = K / block_k;
    int block_bytes    = (int)(sizeof(float) * 2 + block_k / 2);

    for (int r = 0; r < M; ++r) {
        const unsigned char* row = A_q4 + r * blocks_per_row * block_bytes;
        for (int c = 0; c < N; ++c) {
            float sum = 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                const unsigned char* blk = row + b * block_bytes;
                float d, m;
                memcpy(&d, blk, sizeof(float));
                memcpy(&m, blk + sizeof(float), sizeof(float));
                const unsigned char* qs = blk + sizeof(float) * 2;
                for (int i = 0; i < block_k; ++i) {
                    float va = deq_nibble(qs, i, d, m);
                    sum += va * B[(b * block_k + i) * N + c];
                }
            }
            C[r * N + c] = alpha * sum + beta * C[r * N + c];
        }
    }
}

// ===========================================================================
// Variant registry — add new versions here
// ===========================================================================
using GemmQ4Func = void (*)(const unsigned char*, const float*, float*,
                            int, int, int, int, float, float);

struct Variant {
    const char* name;
    GemmQ4Func  func;
};

static const Variant kVariants[] = {
    {"v0", launch_gemm_q4_1_v0},
    {"v1", launch_gemm_q4_1_v1},
    {"v2", launch_gemm_q4_1_v2},   // future
};

// ===========================================================================
// Test fixture
// ===========================================================================
class GemmQ4Test : public ::testing::Test {
protected:
    void run_case(int M, int N, int K, int block_k = 32,
                  float alpha = 1.0f, float beta = 0.0f,
                  float eps = 5.0f) {
        std::vector<float> A_f32(M * K);
        for (size_t i = 0; i < A_f32.size(); ++i)
            A_f32[i] = (float)(rand() % 100) / 50.0f - 1.0f;

        std::vector<float> B_f32(K * N);
        for (size_t i = 0; i < B_f32.size(); ++i)
            B_f32[i] = (float)(rand() % 100) / 50.0f - 1.0f;

        int blocks_per_row = K / block_k;
        int block_bytes    = (int)(sizeof(float) * 2 + block_k / 2);
        std::vector<unsigned char> A_q4(M * blocks_per_row * block_bytes);
        for (int r = 0; r < M; ++r)
            for (int b = 0; b < blocks_per_row; ++b)
                pack_q4_1(&A_f32[r * K + b * block_k], block_k, block_k,
                          &A_q4[r * blocks_per_row * block_bytes + b * block_bytes]);

        std::vector<float> C_cpu(M * N, 0.0f);
        gemm_q4_ref(A_q4.data(), B_f32.data(), C_cpu.data(),
                    M, N, K, block_k, alpha, beta);

        for (auto& [name, func] : kVariants) {
            SCOPED_TRACE(name);
            std::vector<float> C_ocl(M * N, 0.0f);
            func(A_q4.data(), B_f32.data(), C_ocl.data(),
                 M, N, K, block_k, alpha, beta);
            expect_near(C_ocl, C_cpu, eps);
        }
    }
};

// ===========================================================================
// Test cases  —  each runs against ALL registered variant kernels
// ===========================================================================

TEST_F(GemmQ4Test, TinySquare)   { run_case(8,  8,  64); }
TEST_F(GemmQ4Test, Small)        { run_case(16, 16, 128); }
TEST_F(GemmQ4Test, Medium)       { run_case(32, 32, 256); }
TEST_F(GemmQ4Test, RectM)        { run_case(64, 16, 128); }
TEST_F(GemmQ4Test, RectN)        { run_case(16, 64, 128); }
TEST_F(GemmQ4Test, LargeK)       { run_case(16, 16, 512); }
TEST_F(GemmQ4Test, AlphaBeta)    { run_case(16, 16, 128, 32, 2.0f, 0.5f); }
