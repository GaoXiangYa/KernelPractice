#include "device.h"
#include "sgemv.h"

#include <format>
#include <iostream>
#include <string>
#include <vector>

// ===========================================================================
// Variant registry
// ===========================================================================
using SgemvFunc = double (*)(const float*, const float*, float*, int, int,
                             float, float);

struct Variant {
  const char* name;
  SgemvFunc func;
};

static const Variant kVariants[] = {
    // {"v0", sgemv_v0_benchmark},
    {"v1", sgemv_v1_benchmark},
    {"v2", sgemv_v2_benchmark},
    {"v3", sgemv_v3_benchmark},
    {"v4", sgemv_v4_benchmark},
    {"v5", sgemv_v5_benchmark},
};

// ===========================================================================
// Benchmark helper
// ===========================================================================
static void bench_one(SgemvFunc func, const char* label, int M, int N,
                      float alpha, float beta, int warmup, int timed) {
  size_t bytes_A = sizeof(float) * (size_t) M * N;
  size_t bytes_x = sizeof(float) * (size_t) N;
  size_t bytes_y = sizeof(float) * (size_t) M;

  std::vector<float> A(M * N);
  std::vector<float> x(N);
  std::vector<float> y(M, 0.0f);

  for (size_t i = 0; i < M * N; ++i)
    A[i] = (float) (rand()) / (float) RAND_MAX * 2.0f - 1.0f;
  for (size_t i = 0; i < N; ++i)
    x[i] = (float) (rand()) / (float) RAND_MAX * 2.0f - 1.0f;

  // warmup
  for (int i = 0; i < warmup; ++i)
    func(A.data(), x.data(), y.data(), M, N, alpha, beta);

  // timed — device profiling time (launch_profiled)
  double elapsed_ms = 0.0;
  for (int i = 0; i < timed; ++i) {
    elapsed_ms += func(A.data(), x.data(), y.data(), M, N, alpha, beta);
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

  std::cout << std::format("sgemv  benchmark  (warmup={}, iters={})\n\n",
                           warmup, timed);
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
      bench_one(v.func, label.c_str(), p.M, p.N, alpha, beta, warmup, timed);
    }
  }

  std::cout << std::endl;
  return 0;
}
