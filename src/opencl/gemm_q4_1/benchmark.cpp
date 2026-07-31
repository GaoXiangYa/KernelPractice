#include "gemm_q4_1.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <iostream>
#include <vector>

// ===========================================================================
// Pack Q4_1 from float source
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
// Variant registry — mirrors test.cpp
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
int main() {
    constexpr int   block_k = 32;
    constexpr float alpha   = 1.0f, beta = 0.0f;
    constexpr int   warmup  = 3;
    constexpr int   timed   = 20;

    std::cout << std::format(
        "gemm_q4_1  benchmark  (block_k={}, warmup={}, iters={})\n\n",
        block_k, warmup, timed);
    std::cout << "variant      M    N    K        ms     GFLOPS\n";
    std::cout << "--------   ---  ---  ---  --------  -------\n";

    struct Prob { int M, N, K; const char* tag; };
    Prob probs[] = {
        { 128,  128,  128, "tiny"},
        { 256,  256,  256, "S   "},
        { 512,  512,  512, "M   "},
        {1024, 1024, 1024, "L   "},
    };

    for (auto& p : probs) {
        int blocks = p.K / block_k, bytes = sizeof(float)*2 + block_k/2;
        std::vector<unsigned char> A_q4(p.M * blocks * bytes);
        std::vector<float> A_f32(p.M * p.K);
        std::vector<float> B(p.K * p.N, 1.0f);
        std::vector<float> C(p.M * p.N, 0.0f);

        for (int r = 0; r < p.M; ++r) {
            for (int i = 0; i < p.K; ++i)
                A_f32[r*p.K + i] = (float)(rand()%100)/10.0f;
            for (int b = 0; b < blocks; ++b)
                pack_q4_1(&A_f32[r*p.K + b*block_k], block_k, block_k,
                          &A_q4[r * blocks * bytes + b * bytes]);
        }

        for (auto& v : kVariants) {
            // warmup
            for (int i = 0; i < warmup; ++i)
                v.func(A_q4.data(), B.data(), C.data(),
                       p.M, p.N, p.K, block_k, alpha, beta);

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < timed; ++i)
                v.func(A_q4.data(), B.data(), C.data(),
                       p.M, p.N, p.K, block_k, alpha, beta);
            auto t1 = std::chrono::high_resolution_clock::now();

            double s = std::chrono::duration<double>(t1 - t0).count() / timed;
            double flops   = 2.0 * p.M * p.N * p.K;
            double gflops  = flops / s / 1e9;

            std::string label = std::format("{}_{}", v.name, p.tag);
            std::cout << std::format(
                "  {}  | {:4d} {:4d} {:4d} | {:8.3f} ms | {:7.1f} GFLOPS\n",
                label, p.M, p.N, p.K, s * 1e3, gflops);
        }
    }

    std::cout << std::endl;
    return 0;
}
