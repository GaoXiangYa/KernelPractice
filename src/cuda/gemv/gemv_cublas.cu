#include "gemv.h"
#include "util.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

void cublas_gemv_fp32(float *mat_a, float *vec_x, float *vec_y, const int m,
                      const int n, const float alpha, const float beta) {

  float *dA, *dX, *dY;

  cudaMalloc(&dA, m * n * sizeof(float));
  cudaMalloc(&dX, n * sizeof(float));
  cudaMalloc(&dY, m * sizeof(float));

  cudaMemcpy(dA, mat_a, m * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dX, vec_x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dY, vec_y, m * sizeof(float), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  // cuBLAS 使用 ColumnMajor，所以我们把 RowMajor 的 A 作为转置矩阵输入
  // A: m × n (RowMajor)
  // cuBLAS: treat A^T: n × m (ColumnMajor)
  cublasSgemv(handle,
              CUBLAS_OP_T,   // A is treated as transposed
              n,             // cuBLAS A rows    = original n
              m,             // cuBLAS A cols    = original m
              &alpha, dA, n, // lda = leading dim = n
              dX, 1,         // x stride
              &beta, dY, 1); // y stride

  cudaMemcpy(vec_y, dY, m * sizeof(float), cudaMemcpyDeviceToHost);

  cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dX);
  cudaFree(dY);
}

void benchmark_cublas_gemv_fp32(std::ofstream &file, const int m, const int n) {
  constexpr float alpha = 1.0f, beta = 0.0f;
  std::vector<float> mat_a(m * n, 0.0f);
  std::vector<float> vec_x(n, 0.0f);
  std::vector<float> vec_y(m, 0.0f);

  init_random(mat_a, -1.0f, 1.0f);
  init_random(vec_x, -1.0f, 1.0f);
  init_random(vec_y, -1.0f, 1.0f);

  float *dA, *dX, *dY;

  cudaMalloc(&dA, m * n * sizeof(float));
  cudaMalloc(&dX, n * sizeof(float));
  cudaMalloc(&dY, m * sizeof(float));

  cudaMemcpy(dA, mat_a.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dX, vec_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dY, vec_y.data(), m * sizeof(float), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  constexpr int warmup = 10;
  for (int i = 0; i < warmup; ++i) {
    cublasSgemv(handle,
              CUBLAS_OP_T,   // A is treated as transposed
              n,             // cuBLAS A rows    = original n
              m,             // cuBLAS A cols    = original m
              &alpha, dA, n, // lda = leading dim = n
              dX, 1,         // x stride
              &beta, dY, 1); // y stride
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  constexpr int iters = 10;

  for (int i = 0; i < iters; i++) {
    cublasSgemv(handle,
              CUBLAS_OP_T,   // A is treated as transposed
              n,             // cuBLAS A rows    = original n
              m,             // cuBLAS A cols    = original m
              &alpha, dA, n, // lda = leading dim = n
              dX, 1,         // x stride
              &beta, dY, 1); // y stride
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  ms /= iters;

  // Compute GFLOPS
  double flops = 2.0 * m * n;
  double gflops = flops / (ms * 1e6);

  file << "CUBLAS_GEMV: M = " << m << " N = " << n << " -> " << ms << " ms "
       << gflops << " GFlops\n";
  cudaMemcpy(vec_y.data(), dY, m * sizeof(float), cudaMemcpyDeviceToHost);

  cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dX);
  cudaFree(dY);
}
