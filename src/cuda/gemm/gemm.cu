#include "gemm_v0_kernel.cuh"
#include "gemm_v1_kernel.cuh"
#include "gemm_v2_kernel.cuh"
#include "gemm_v3_kernel.cuh"
#include "util.h"

void gemm_v0(const float* a, const float* b, float* c, int M, int N, int K) {
  int lda = K, ldb = N, ldc = N;

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, M * N * sizeof(float)));

  CHECK_CUDA(
      cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

  constexpr int BM = 16;
  constexpr int BN = 16;
  constexpr int BK = 8;

  dim3 block(BM, BN);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  gemm_v0_kernel<BM, BN, BK>
      <<<grid, block>>>(dev_a, dev_b, dev_c, M, N, K, lda, ldb, ldc);

  CHECK_CUDA(
      cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}

void gemm_v1(const float* a, const float* b, float* c, int M, int N, int K) {
  int lda = K, ldb = N, ldc = N;

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, M * N * sizeof(float)));

  CHECK_CUDA(
      cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 32;

  constexpr int kThreadCountX = 32;
  constexpr int kThreadCountY = 32;

  dim3 block(kThreadCountX, kThreadCountY);
  dim3 grid((N + kThreadCountX - 1) / kThreadCountX,
            (M + kThreadCountY - 1) / kThreadCountY);

  gemm_v1_kernel<BM, BN, BK>
      <<<grid, block>>>(dev_a, dev_b, dev_c, M, N, K, lda, ldb, ldc);

  CHECK_CUDA(
      cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}

void gemm_v2(const float* a, const float* b, float* c, int M, int N, int K) {
  int lda = K, ldb = N, ldc = N;

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, M * N * sizeof(float)));

  CHECK_CUDA(
      cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

  constexpr int BM = 256;
  constexpr int BN = 64;
  constexpr int BK = 32;
  constexpr int kMicroSize = 4;

  constexpr int kThreadCountX = 32;
  constexpr int kThreadCountY = 32 * kMicroSize;

  dim3 block(kThreadCountX, kThreadCountY);
  dim3 grid((N + kThreadCountX - 1) / kThreadCountX,
            (M + (kThreadCountY * kMicroSize) - 1) /
                (kThreadCountY * kMicroSize));

  gemm_v2_kernel<BM, BN, BK, kMicroSize, 1>
      <<<grid, block>>>(dev_a, dev_b, dev_c, M, N, K, lda, ldb, ldc);

  CHECK_CUDA(
      cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}

void gemm_v3(const float* a, const float* b, float* c, int M, int N, int K) {
  int lda = K, ldb = N, ldc = N;

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, M * N * sizeof(float)));

  CHECK_CUDA(
      cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

  constexpr int BM = 256;
  constexpr int BN = 64;
  constexpr int BK = 32;
  constexpr int kMicroSize = 4;

  constexpr int kThreadCountX = 32;
  constexpr int kThreadCountY = 32 * kMicroSize;

  dim3 block(kThreadCountX, kThreadCountY);
  dim3 grid((N + kThreadCountX - 1) / kThreadCountX,
            (M + (kThreadCountY * kMicroSize) - 1) /
                (kThreadCountY * kMicroSize));

  gemm_v3_kernel<BM, BN, BK, kMicroSize, 1>
      <<<grid, block>>>(dev_a, dev_b, dev_c, M, N, K, lda, ldb, ldc);

  CHECK_CUDA(
      cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}
