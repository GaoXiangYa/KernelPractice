#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "benchmark.cuh"
#include "gemm_v0_kernel.cuh"
#include "gemm_v1_kernel.cuh"
#include "gemm_v2_kernel.cuh"
#include "gemm_v3_kernel.cuh"
#include "util.h"

static void run_benchmark(int M, int N, int K) {
  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c(M * N, 0.0f);

  for (auto& v : a)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;
  for (auto& v : b)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, M * N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_a, a.data(), M * K * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dev_b, b.data(), K * N * sizeof(float),
                        cudaMemcpyHostToDevice));

  int lda = K, ldb = N, ldc = N;

  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 64;

  dim3 block(BM, BN);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  // 2 * M * N * K FLOPs (one mul + one add per inner loop)
  double flops = 2.0 * M * N * K;
  double bytes = (M * K + K * N + M * N) * sizeof(float);

  char name[128];
  snprintf(name, sizeof(name), "gemm_v0 [M=%d,N=%d,K=%d]", M, N, K);

  std::cout << "\n";
  benchmarkKernel(name, grid, block, flops, bytes, 10,
                  gemm_v0_kernel<BM, BN, BK>, dev_a, dev_b, dev_c, M, N, K, lda,
                  ldb, ldc);

  CHECK_CUDA(cudaMemcpy(c.data(), dev_c, M * N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}

static void run_benchmark_v1(int M, int N, int K) {
  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c(M * N, 0.0f);

  for (auto& v : a)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;
  for (auto& v : b)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, M * N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_a, a.data(), M * K * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dev_b, b.data(), K * N * sizeof(float),
                        cudaMemcpyHostToDevice));

  int lda = K, ldb = N, ldc = N;

  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 32;

  constexpr int kThreadCountX = 32;
  constexpr int kThreadCountY = 32;

  dim3 block(kThreadCountX, kThreadCountY);
  dim3 grid((N + kThreadCountX - 1) / kThreadCountX,
            (M + kThreadCountY - 1) / kThreadCountY);

  // 2 * M * N * K FLOPs (one mul + one add per inner loop)
  double flops = 2.0 * M * N * K;
  double bytes = (M * K + K * N + M * N) * sizeof(float);

  char name[128];
  snprintf(name, sizeof(name), "gemm_v1 [M=%d,N=%d,K=%d]", M, N, K);

  std::cout << "\n";
  benchmarkKernel(name, grid, block, flops, bytes, 1,
                  gemm_v1_kernel<BM, BN, BK>, dev_a, dev_b, dev_c, M, N, K, lda,
                  ldb, ldc);

  CHECK_CUDA(cudaMemcpy(c.data(), dev_c, M * N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}

static void run_benchmark_v2(int M, int N, int K) {
  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c(M * N, 0.0f);

  for (auto& v : a)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;
  for (auto& v : b)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, M * N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_a, a.data(), M * K * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dev_b, b.data(), K * N * sizeof(float),
                        cudaMemcpyHostToDevice));

  int lda = K, ldb = N, ldc = N;

  constexpr int BM = 256;
  constexpr int BN = 64;
  constexpr int BK = 32;
  constexpr int kMicroSize = 4;

  constexpr int kThreadCountX = 32;
  constexpr int kThreadCountY = 32;

  dim3 block(kThreadCountX, kThreadCountY);
  dim3 grid((N + kThreadCountX - 1) / kThreadCountX,
            (M + (kThreadCountY * kMicroSize) - 1) /
                (kThreadCountY * kMicroSize));

  // 2 * M * N * K FLOPs (one mul + one add per inner loop)
  double flops = 2.0 * M * N * K;
  double bytes = (M * K + K * N + M * N) * sizeof(float);

  char name[128];
  snprintf(name, sizeof(name), "gemm_v2 [M=%d,N=%d,K=%d]", M, N, K);

  std::cout << "\n";
  benchmarkKernel(name, grid, block, flops, bytes, 1,
                  gemm_v2_kernel<BM, BN, BK, kMicroSize, 1>, dev_a, dev_b,
                  dev_c, M, N, K, lda, ldb, ldc);

  CHECK_CUDA(cudaMemcpy(c.data(), dev_c, M * N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}

static void run_benchmark_v3(int M, int N, int K) {
  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c(M * N, 0.0f);

  for (auto& v : a)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;
  for (auto& v : b)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, M * N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_a, a.data(), M * K * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dev_b, b.data(), K * N * sizeof(float),
                        cudaMemcpyHostToDevice));

  int lda = K, ldb = N, ldc = N;

  constexpr int BM = 256;
  constexpr int BN = 64;
  constexpr int BK = 32;
  constexpr int kMicroSize = 4;

  constexpr int kThreadCountX = 32;
  constexpr int kThreadCountY = 32;

  dim3 block(kThreadCountX, kThreadCountY);
  dim3 grid((N + kThreadCountX - 1) / kThreadCountX,
            (M + (kThreadCountY * kMicroSize) - 1) /
                (kThreadCountY * kMicroSize));

  // 2 * M * N * K FLOPs (one mul + one add per inner loop)
  double flops = 2.0 * M * N * K;
  double bytes = (M * K + K * N + M * N) * sizeof(float);

  char name[128];
  snprintf(name, sizeof(name), "gemm_v2 [M=%d,N=%d,K=%d]", M, N, K);

  std::cout << "\n";
  benchmarkKernel(name, grid, block, flops, bytes, 1,
                  gemm_v3_kernel<BM, BN, BK, kMicroSize, 1>, dev_a, dev_b,
                  dev_c, M, N, K, lda, ldb, ldc);

  CHECK_CUDA(cudaMemcpy(c.data(), dev_c, M * N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}
int main() {
  std::cout << "=== CUDA GEMM Benchmark ===\n";

  // std::cout << "\n--- gemm_v0 ---\n";
  // run_benchmark(2048, 2048, 2048);
  // run_benchmark(1024, 16, 256);

  // std::cout << "\n--- gemm_v1 ---\n";
  // run_benchmark_v1(2048, 2048, 2048);
  // // run_benchmark_v1(1024, 16, 256);

  std::cout << "\n--- gemm_v2 ---\n";
  run_benchmark_v2(2048, 2048, 2048);
  // run_benchmark_v2(1024, 16, 256);

  std::cout << "\n--- gemm_v3 ---\n";
  run_benchmark_v3(2048, 2048, 2048);
  // run_benchmark_v2(1024, 16, 256);

  return 0;
}
