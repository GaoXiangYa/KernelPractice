#include "gemm_v0_kernel.cuh"
#include "gemm_v1_kernel.cuh"
#include "gemm_v2_kernel.cuh"
#include "gemm_v3_kernel.cuh"
#include "gemm_v4_kernel.cuh"
#include "gemm_v5_kernel.cuh"
#include "gemm_v6_kernel.cuh"
#include "gemm_v7_kernel.cuh"
#include "gemm_v8_kernel.cuh"
#include "gemm_v9_kernel.cuh"
#include "kernel_common.cuh"
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

  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;

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

  constexpr int BM = 32;
  constexpr int BN = 32;
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

  constexpr int BM = 128;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int TM = 4;
  constexpr int TN = 1;

  constexpr int kThreadCountX = 32;
  constexpr int kThreadCountY = 32;

  dim3 block(kThreadCountX, kThreadCountY);
  dim3 grid((N + (kThreadCountX * TN) - 1) / (kThreadCountX * TN),
            (M + (kThreadCountY * TM) - 1) / (kThreadCountY * TM));

  gemm_v2_kernel<BM, BN, BK, TM, TN>
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

  constexpr int BM = 128;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int TM = 4;
  constexpr int TN = 1;

  constexpr int kThreadCountX = 32;
  constexpr int kThreadCountY = 32;

  dim3 block(kThreadCountX, kThreadCountY);
  dim3 grid((N + (kThreadCountX * TN) - 1) / (kThreadCountX * TN),
            (M + (kThreadCountY * TM) - 1) / (kThreadCountY * TM));

  gemm_v3_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(dev_a, dev_b, dev_c, M, N, K, lda, ldb, ldc);

  CHECK_CUDA(
      cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}

void gemm_v4(const float* a, const float* b, float* c, int M, int N, int K) {
  int lda = K, ldb = N, ldc = N;

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, M * N * sizeof(float)));

  CHECK_CUDA(
      cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 32;
  constexpr int TM = 4;
  constexpr int TN = 4;

  constexpr int kThreadCountX = 32;
  constexpr int kThreadCountY = 32;

  dim3 block(kThreadCountX, kThreadCountY);
  dim3 grid((N + (kThreadCountX * TN) - 1) / (kThreadCountX * TN),
            (M + (kThreadCountY * TM) - 1) / (kThreadCountY * TM));

  gemm_v4_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(dev_a, dev_b, dev_c, M, N, K, lda, ldb, ldc);

  CHECK_CUDA(
      cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}

void gemm_v5(const float* a, const float* b, float* c, int M, int N, int K) {
  int lda = K, ldb = N, ldc = N;

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, M * N * sizeof(float)));

  CHECK_CUDA(
      cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16;
  constexpr int TM = 8;
  constexpr int TN = 8;

  constexpr int kThreadCountX = 16;
  constexpr int kThreadCountY = 16;

  dim3 block(kThreadCountX, kThreadCountY);
  dim3 grid((N + (kThreadCountX * TN) - 1) / (kThreadCountX * TN),
            (M + (kThreadCountY * TM) - 1) / (kThreadCountY * TM));

  gemm_v5_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(dev_a, dev_b, dev_c, M, N, K, lda, ldb, ldc);

  CHECK_CUDA(
      cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}

void gemm_v6(const float* a, const float* b, float* c, int M, int N, int K) {
  int lda = K, ldb = N, ldc = N;

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, M * N * sizeof(float)));

  CHECK_CUDA(
      cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 32;
  constexpr int TM = 8;
  constexpr int TN = 4;

  constexpr int kThreadCountX = 32;
  constexpr int kThreadCountY = 16;

  dim3 block(kThreadCountX, kThreadCountY);
  dim3 grid((N + (kThreadCountX * TN) - 1) / (kThreadCountX * TN),
            (M + (kThreadCountY * TM) - 1) / (kThreadCountY * TM));

  //   cudaFuncSetAttribute(gemm_v6_kernel<BM, BN, BK, TM, TN>,
  //                        cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

  gemm_v6_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(dev_a, dev_b, dev_c, M, N, K, lda, ldb, ldc);

  CHECK_CUDA(
      cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}

void gemm_v7(const float* a, const float* b, float* c, int M, int N, int K) {
  int lda = K, ldb = N, ldc = N;

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, M * N * sizeof(float)));

  CHECK_CUDA(
      cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 32;
  constexpr int TM = 8;
  constexpr int TN = 4;

  constexpr int kThreadCountX = 32;
  constexpr int kThreadCountY = 16;

  dim3 block(kThreadCountX, kThreadCountY);
  dim3 grid((N + (kThreadCountX * TN) - 1) / (kThreadCountX * TN),
            (M + (kThreadCountY * TM) - 1) / (kThreadCountY * TM));

  //   cudaFuncSetAttribute(gemm_v6_kernel<BM, BN, BK, TM, TN>,
  //                        cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

  gemm_v7_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(dev_a, dev_b, dev_c, M, N, K, lda, ldb, ldc);

  CHECK_CUDA(
      cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}

void gemm_v8(const float* a, const float* b, float* c, int M, int N, int K) {
  int lda = K, ldb = N, ldc = N;

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, M * N * sizeof(float)));

  CHECK_CUDA(
      cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

  using V8 = GemmConfig<128, 128, 32, 32, 8, 4>;
  dim3 block(V8::THREADS);
  dim3 grid((N + V8::BLOCK_TILE_N - 1) / V8::BLOCK_TILE_N,
            (M + V8::BLOCK_TILE_M - 1) / V8::BLOCK_TILE_M);
  gemm_v8_kernel<V8>
      <<<grid, block>>>(dev_a, dev_b, dev_c, M, N, K, lda, ldb, ldc);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(
      cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}

void gemm_v9(const float* a, const float* b, float* c, int M, int N, int K) {
  int lda = K, ldb = N, ldc = N;

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, M * N * sizeof(float)));

  CHECK_CUDA(
      cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

  using V9 = GemmConfig<128, 128, 32, 32, 8, 4>;
  dim3 block(V9::THREADS);
  dim3 grid((N + V9::BLOCK_TILE_N - 1) / V9::BLOCK_TILE_N,
            (M + V9::BLOCK_TILE_M - 1) / V9::BLOCK_TILE_M);
  v9::gemm_v9_kernel<V9>
      <<<grid, block>>>(dev_a, dev_b, dev_c, M, N, K, lda, ldb, ldc);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(
      cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
}