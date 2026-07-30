#include "benchmark.cuh"
#include "gemm.cuh"
#include "gemm_v0_kernel.cuh"
#include "gemm_v1_kernel.cuh"
#include "gemm_v2_kernel.cuh"
#include "gemm_v3_kernel.cuh"
#include "gemm_v4_kernel.cuh"
#include "gemm_v5_kernel.cuh"
#include "gemm_v6_kernel.cuh"
#include "util.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---- per-version kernel launch helpers (no memcpy, pure kernel timing) ----

static void launch_v0(const float* da, const float* db, float* dc, int M, int N,
                      int K) {
  constexpr int BM = 32, BN = 32, BK = 32;
  int lda = K, ldb = N, ldc = N;
  dim3 block(BM, BN);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_v0_kernel<BM, BN, BK>
      <<<grid, block>>>(da, db, dc, M, N, K, lda, ldb, ldc);
}

static void launch_v1(const float* da, const float* db, float* dc, int M, int N,
                      int K) {
  constexpr int BM = 32, BN = 32, BK = 32;
  constexpr int TX = 32, TY = 32;
  int lda = K, ldb = N, ldc = N;
  dim3 block(TX, TY);
  dim3 grid((N + TX - 1) / TX, (M + TY - 1) / TY);
  gemm_v1_kernel<BM, BN, BK>
      <<<grid, block>>>(da, db, dc, M, N, K, lda, ldb, ldc);
}

static void launch_v2(const float* da, const float* db, float* dc, int M, int N,
                      int K) {
  constexpr int BM = 128, BN = 32, BK = 32, TM = 4, TN = 1;
  constexpr int TX = 32, TY = 32;
  int lda = K, ldb = N, ldc = N;
  dim3 block(TX, TY);
  dim3 grid((N + TX * TN - 1) / (TX * TN), (M + TY * TM - 1) / (TY * TM));
  gemm_v2_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(da, db, dc, M, N, K, lda, ldb, ldc);
}

static void launch_v3(const float* da, const float* db, float* dc, int M, int N,
                      int K) {
  constexpr int BM = 128, BN = 32, BK = 32, TM = 4, TN = 1;
  constexpr int TX = 32, TY = 32;
  int lda = K, ldb = N, ldc = N;
  dim3 block(TX, TY);
  dim3 grid((N + TX * TN - 1) / (TX * TN), (M + TY * TM - 1) / (TY * TM));
  gemm_v3_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(da, db, dc, M, N, K, lda, ldb, ldc);
}

static void launch_v4(const float* da, const float* db, float* dc, int M, int N,
                      int K) {
  constexpr int BM = 128, BN = 128, BK = 32, TM = 4, TN = 4;
  constexpr int TX = 32, TY = 32;
  int lda = K, ldb = N, ldc = N;
  dim3 block(TX, TY);
  dim3 grid((N + TX * TN - 1) / (TX * TN), (M + TY * TM - 1) / (TY * TM));
  gemm_v4_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(da, db, dc, M, N, K, lda, ldb, ldc);
}

static void launch_v5(const float* da, const float* db, float* dc, int M, int N,
                      int K) {
  constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
  constexpr int TX = 16, TY = 16;
  int lda = K, ldb = N, ldc = N;
  dim3 block(TX, TY);
  dim3 grid((N + TX * TN - 1) / (TX * TN), (M + TY * TM - 1) / (TY * TM));
  gemm_v5_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(da, db, dc, M, N, K, lda, ldb, ldc);
}

static void launch_v6(const float* da, const float* db, float* dc, int M, int N,
                      int K) {
  constexpr int BM = 256, BN = 128, BK = 32, TM = 8, TN = 8;
  constexpr int TX = 16, TY = 32;
  int lda = K, ldb = N, ldc = N;
  dim3 block(TX, TY);
  dim3 grid((N + TX * TN - 1) / (TX * TN), (M + TY * TM - 1) / (TY * TM));
  gemm_v6_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(da, db, dc, M, N, K, lda, ldb, ldc);
}
// ---- timing helper ----

static double
bench(const char* name,
      void (*launch)(const float*, const float*, float*, int, int, int),
      const float* da, const float* db, float* dc, int M, int N, int K) {
  // warmup
  launch(da, db, dc, M, N, K);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int r = 0; r < 1; ++r)
    launch(da, db, dc, M, N, K);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  ms /= 10.0f;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double gflops = (2.0 * M * N * K) / (ms / 1000.0) / 1e9;
  std::printf("%s,%d,%d,%d,%.2f\n", name, M, N, K, gflops);
  return gflops;
}

// ---- dispatch table ----

struct KernelEntry {
  const char* name;
  void (*launch)(const float*, const float*, float*, int, int, int);
};

static const KernelEntry kKernels[] = {{"v0", launch_v0}, {"v1", launch_v1},
                                       {"v2", launch_v2}, {"v3", launch_v3},
                                       {"v4", launch_v4}, {"v5", launch_v5},
                                       {"v6", launch_v6}};

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
  std::printf("version,M,N,K,gflops\n");

  constexpr int bg = 6144;
  constexpr int ed = 6144;
  constexpr int step = 256;
  for (int size = bg; size <= ed; size += step) {
    int M = size, N = size, K = size;

    std::vector<float> ha(M * K), hb(K * N);
    for (auto& v : ha)
      v = float(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (auto& v : hb)
      v = float(rand()) / RAND_MAX * 2.0f - 1.0f;

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CHECK_CUDA(cudaMalloc(&da, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&db, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dc, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(da, ha.data(), M * K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(db, hb.data(), K * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    for (const auto& ke : kKernels) {
      if (enabled(ke, argc, argv))
        bench(ke.name, ke.launch, da, db, dc, M, N, K);
    }

    CHECK_CUDA(cudaFree(da));
    CHECK_CUDA(cudaFree(db));
    CHECK_CUDA(cudaFree(dc));
  }
  return 0;
}
