#include "device.h"
#include "gemm.h"

#include <format>
#include <iostream>
#include <string>
#include <vector>

// ===========================================================================
// Variant registry
// ===========================================================================
using GemmFunc = double (*)(const float*, const float*, float*, int, int, int,
                            float, float);

struct Variant {
  const char* name;
  GemmFunc func;
};

static const Variant kVariants[] = {
    {"v0", gemm_v0_benchmark}, {"v1", gemm_v1_benchmark},
    {"v2", gemm_v2_benchmark}, {"v3", gemm_v3_benchmark},
    {"v4", gemm_v4_benchmark}, {"v5", gemm_v5_benchmark},
    {"v6", gemm_v6_benchmark}, {"v7", gemm_v7_benchmark},
    {"v8", gemm_v8_benchmark}, {"v9", gemm_v9_benchmark},
};

// ===========================================================================
// Benchmark helper
// ===========================================================================
static void bench_one(GemmFunc func, const char* label, int M, int N, int K,
                      float alpha, float beta, int warmup, int timed) {
  size_t bytes_A = sizeof(float) * (size_t) M * K;
  size_t bytes_B = sizeof(float) * (size_t) K * N;
  size_t bytes_C = sizeof(float) * (size_t) M * N;

  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C(M * N, 0.0f);

  for (size_t i = 0; i < M * K; ++i)
    A[i] = (float) (rand()) / (float) RAND_MAX * 2.0f - 1.0f;
  for (size_t i = 0; i < K * N; ++i)
    B[i] = (float) (rand()) / (float) RAND_MAX * 2.0f - 1.0f;

  // warmup
  for (int i = 0; i < warmup; ++i)
    func(A.data(), B.data(), C.data(), M, N, K, alpha, beta);

  // timed — host clock (launch functions do blocking readback)
  // auto t0 = std::chrono::high_resolution_clock::now();
  double elapsed_ms = 0.0;
  for (int i = 0; i < timed; ++i) {
    elapsed_ms += func(A.data(), B.data(), C.data(), M, N, K, alpha, beta);
  }

  double flops = 2.0 * (double) M * (double) N * (double) K;
  double gflops = flops / elapsed_ms / 1e6;
  double bytes = (double) (bytes_A + bytes_B + bytes_C);
  double bw = bytes / elapsed_ms / 1e6;

  std::cout << std::format(
      "{} | {:4d} {:4d} {:4d} | {:8.3f} ms | {:8.3f} GFLOPS | {:8.3f} GB/s\n",
      label, M, N, K, elapsed_ms, gflops, bw);
}

// ===========================================================================
int main() {
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int warmup = 3;
  constexpr int timed = 20;

  std::cout << std::format("gemm  benchmark  (warmup={}, iters={})\n\n", warmup,
                           timed);
  std::cout << "variant    M    N    K        ms     GFLOPS    GB/s\n";
  std::cout << "--------  ---  ---  ---  --------  -------  ------\n";

  struct Problem {
    int M, N, K;
    const char* tag;
  };
  Problem probs[] = {
      {128, 128, 128, "tiny  "},    {256, 256, 256, "small "},
      {512, 512, 512, "medium"},    {1024, 1024, 1024, "large "},
      {4096, 4096, 4096, "rect  "},
  };

  for (auto& p : probs) {
    for (auto& v : kVariants) {
      std::string label = std::format("{}_{}", v.name, p.tag);
      bench_one(v.func, label.c_str(), p.M, p.N, p.K, alpha, beta, warmup,
                timed);
    }
  }

  std::cout << std::endl;
  return 0;
}
