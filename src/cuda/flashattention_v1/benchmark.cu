#include "flashattention_v1.h"
#include "util.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---- shape table (IMPLEMENTATION.md §8) ----
struct Shape {
  int B, H, N, d;
};

static const Shape kShapes[] = {
    {1, 1, 1024, 64},
    {1, 8, 2048, 64},
    {1, 16, 4096, 128},
    {4, 16, 2048, 64},
};

// ---- per-version launch helpers (public API, incl. H2D/D2H) ----
// TODO(M1): direct kernel timing without copies, see gemm benchmark launch_vN

struct KernelEntry {
  const char* name;
  void (*fn)(const float*, const float*, const float*, float*, int, int, int,
             int, bool);
};

static const KernelEntry kKernels[] = {
    {"v0", flash_attn_v0},
    {"v1", flash_attn_v1},
};

static bool enabled(const KernelEntry& e, int argc, char** argv) {
  if (argc <= 1)
    return true;  // no args → run all
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], e.name) == 0)
      return true;
  return false;
}

// ---- main ----
int main(int argc, char** argv) {
  std::printf("version,B,H,N,d,causal,gflops,gbps\n");

  for (const auto& s : kShapes) {
    const int total = s.B * s.H * s.N * s.d;
    std::vector<float> q(total), k(total), v(total), o(total, 0.0f);
    for (auto& x : q) x = float(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (auto& x : k) x = float(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (auto& x : v) x = float(rand()) / RAND_MAX * 2.0f - 1.0f;

    for (int causal_i = 0; causal_i < 2; ++causal_i) {
      bool causal = causal_i == 1;
      for (const auto& ke : kKernels) {
        if (!enabled(ke, argc, argv))
          continue;

        // warmup
        ke.fn(q.data(), k.data(), v.data(), o.data(), s.B, s.H, s.N, s.d, causal);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        constexpr int kRepeats = 10;
        for (int r = 0; r < kRepeats; ++r)
          ke.fn(q.data(), k.data(), v.data(), o.data(), s.B, s.H, s.N, s.d, causal);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= (float)kRepeats;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // statistics (IMPLEMENTATION.md §8)
        double flops = 4.0 * s.B * s.H * s.N * s.N * s.d;
        if (causal) flops *= 0.5;
        // bytes (fp32, non-causal): (B·H·N·d·4)·3 + (B·H·N²·d·8)/B_r, B_r=64
        // TODO(M3): refine bytes formula
        // Note: B*H*N*N*d overflows 32-bit at 2048/4096; cast to double before multiply
        double bytes = (double)s.B * s.H * s.N * s.d * 4.0 * 3 +
                       (double)s.B * s.H * s.N * s.N * s.d * 8.0 / 64.0;
        if (causal) bytes *= 0.5;

        double gflops = flops / (ms / 1000.0) / 1e9;
        double gbps = bytes / (ms / 1000.0) / 1e9;
        std::printf("%s,%d,%d,%d,%d,%d,%.2f,%.2f\n", ke.name, s.B, s.H, s.N,
                    s.d, causal ? 1 : 0, gflops, gbps);
      }
    }
  }
  return 0;
}
