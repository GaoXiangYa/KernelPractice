#include "gemv.h"
#include "util.h"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

void cutlass_gemv_fp32(float *mat_a, float *vec_x, float *vec_y, const int m,
                       const int n, const float alpha, const float beta) {

  float *dev_mat_a = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_mat_a, m * n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_mat_a, mat_a, m * n * sizeof(float),
                        cudaMemcpyHostToDevice));

  float *dev_vec_x = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_vec_x, n * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(dev_vec_x, vec_x, n * sizeof(float), cudaMemcpyHostToDevice));

  float *dev_vec_y = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_vec_y, m * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(dev_vec_y, vec_y, m * sizeof(float), cudaMemcpyHostToDevice));

  using Gemv =
      cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float,
                                  cutlass::layout::ColumnMajor, float,
                                  cutlass::layout::ColumnMajor, float>;

  Gemv gemv_op;

  Gemv::Arguments args({m, 1, n},      // M × N × K
                       {dev_mat_a, n}, // A pointer + lda
                       {dev_vec_x, n}, // B pointer (as Nx1 matrix)
                       {dev_vec_y, m}, // C pointer (Mx1)
                       {dev_vec_y, m}, // D pointer (output)
                       {alpha, beta}   // alpha, beta
  );

  cutlass::Status status = gemv_op(args);

  if (status != cutlass::Status::kSuccess) {
    printf("CUTLASS GEMV failed\n");
  }

  CHECK_CUDA(
      cudaMemcpy(vec_y, dev_vec_y, m * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_mat_a));
  CHECK_CUDA(cudaFree(dev_vec_x));
  CHECK_CUDA(cudaFree(dev_vec_y));
}

void benchmark_cutlass_gemv_fp32(std::ofstream &file, const int m,
                                 const int n) {
  constexpr float alpha = 1.0f, beta = 0.0f;
  std::vector<float> mat_a(m * n, 0.0f);
  std::vector<float> vec_x(n, 0.0f);
  std::vector<float> vec_y(m, 0.0f);

  init_random(mat_a, -1.0f, 1.0f);
  init_random(vec_x, -1.0f, 1.0f);
  init_random(vec_y, -1.0f, 1.0f);

  float *dev_mat_a = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_mat_a, m * n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_mat_a, mat_a.data(), m * n * sizeof(float),
                        cudaMemcpyHostToDevice));

  float *dev_vec_x = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_vec_x, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_vec_x, vec_x.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  float *dev_vec_y = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_vec_y, m * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_vec_y, vec_y.data(), m * sizeof(float),
                        cudaMemcpyHostToDevice));

  using Gemv =
      cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float,
                                  cutlass::layout::ColumnMajor, float,
                                  cutlass::layout::ColumnMajor, float>;

  Gemv gemv_op;

  Gemv::Arguments args({m, 1, n},      // M × N × K
                       {dev_mat_a, n}, // A pointer + lda
                       {dev_vec_x, n}, // B pointer (as Nx1 matrix)
                       {dev_vec_y, m}, // C pointer (Mx1)
                       {dev_vec_y, m}, // D pointer (output)
                       {alpha, beta}   // alpha, beta
  );

  constexpr int warmup = 10;
  for (int i = 0; i < warmup; ++i) {
    gemv_op(args);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  constexpr int iters = 10;

  for (int i = 0; i < iters; i++) {
    gemv_op(args);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  ms /= iters;

  // Compute GFLOPS
  double flops = 2.0 * m * n;
  double gflops = flops / (ms * 1e6);

  file << "CUTLASS_GEMV: M = " << m << " N = " << n << " -> " << ms << " ms "
       << gflops << " GFlops\n";

  CHECK_CUDA(cudaMemcpy(vec_y.data(), dev_vec_y, m * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_mat_a));
  CHECK_CUDA(cudaFree(dev_vec_x));
  CHECK_CUDA(cudaFree(dev_vec_y));
}
