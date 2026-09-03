#include "device.h"
#include "sgemv_q4_1.h"

#include <cmath>
#include <cstring>
#include <format>
#include <iostream>
#include <string>
#include <vector>

// ===========================================================================
// Q4_1 packing (host) — same convention as gemm_q4_1
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
  memcpy(dst, &d, sizeof(float));
  dst += sizeof(float);
  memcpy(dst, &m, sizeof(float));
  dst += sizeof(float);
  for (int i = 0; i < k; i += 2) {
    int q0 = (int) roundf((src[i] - m) / d * 15.0f);
    int q1 = (i + 1 < k) ? (int) roundf((src[i + 1] - m) / d * 15.0f) : 0;
    q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
    q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
    *dst++ = (unsigned char) (q0 | (q1 << 4));
  }
}

// ===========================================================================
// Variant registry
// ===========================================================================
using SgemvQ4Func = double (*)(const unsigned char*, const float*, float*,
                               int, int, int, float, float);

struct Variant {
  const char* name;
  SgemvQ4Func func;
};

static const Variant kVariants[] = {
    {"v0", sgemv_q4_1_v0_benchmark},
};

// ===========================================================================
// Benchmark helper
// ===========================================================================
static void bench_one(SgemvQ4Func func, const char* label, int M, int N,
                      int block_k, float alpha, float beta, int warmup,
                      int timed) {
  int blocks_per_row = N / block_k;
  int block_bytes = (int) (sizeof(float) * 2 + block_k / 2);
  size_t bytes_A = (size_t) M *blocks_per_row * block_bytes;
  size_t bytes_x = sizeof(float) * (size_t) N;
  size_t bytes_y = sizeof(float) * (size_t) M;

  std::vector<float> A_f32(M * N);
  std::vector<float> x(N);
  std::vector<float> y(M, 0.0f);
  std::vector<unsigned char> A_q4(bytes_A);

  for (size_t i = 0; i < A_f32.size(); ++i)
    A_f32[i] = (float) (rand()) / (float) RAND_MAX * 2.0f - 1.0f;
  for (size_t i = 0; i < N; ++i)
    x[i] = (float) (rand()) / (float) RAND_MAX * 2.0f - 1.0f;
  for (int r = 0; r < M; ++r)
    for (int b = 0; b < blocks_per_row; ++b)
      pack_q4_1(&A_f32[r * N + b * block_k], block_k, block_k,
                &A_q4[r * blocks_per_row * block_bytes + b * block_bytes]);

  for (int i = 0; i < warmup; ++i)
    func(A_q4.data(), x.data(), y.data(), M, N, block_k, alpha, beta);

  double elapsed_ms = 0.0;
  for (int i = 0; i < timed; ++i) {
    elapsed_ms += func(A_q4.data(), x.data(), y.data(), M, N, block_k, alpha,
                       beta);
  }

  double flops = 2.0 * (double) M * (double) N;
  double gflops = flops / elapsed_ms / 1e6;
  double bytes = (double) (bytes_A + bytes_x + bytes_y);
  double bw = bytes / elapsed_ms / 1e6;

  std::cout << std::format(
      "{} | {:5d} {:5d} | {:8.3f} ms | {:8.3f} GFLOPS | {:8.3f} GB/s\n",
      label, M, N, elapsed_ms, gflops, bw);
}

// ===========================================================================
int main() {
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int warmup = 3;
  constexpr int timed = 20;

  std::cout << std::format(
      "sgemv_q4_1  benchmark  (warmup={}, iters={})\n\n", warmup, timed);
  std::cout << "variant      M     N        ms     GFLOPS    GB/s\n";
  std::cout << "--------  ----- -----  --------  -------  ------\n";

  struct Problem {
    int M, N;
    const char* tag;
  };
  Problem probs[] = {
      {1024, 1024, "tiny  "},
      {4096, 4096, "small "},
      {4096, 16384, "medium"},
      {16384, 16384, "large "},
  };

  for (auto& p : probs) {
    for (auto& v : kVariants) {
      std::string label = std::format("{}_{}", v.name, p.tag);
      bench_one(v.func, label.c_str(), p.M, p.N, 32, alpha, beta, warmup,
                timed);
    }
  }

  std::cout << std::endl;
  return 0;
}
