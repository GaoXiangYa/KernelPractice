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
}
