#include "gemm.h"
#include <CL/cl.h>
#include <CL/cl2.hpp>
#include <CL/opencl.hpp>
#include <cstddef>
#include "utils.h"

void gemm_v0(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta) {
  const std::string build_options = "-funsafe-max-local-work-size=2";
  OCLKernel ocl_kernel("../src/opencl/gemm/gemm_v0.cl", "gemm_v0_kernel",
                       build_options);
  const int kGlobalSizeM = M;
  const int kGlobalSizeN = N;

  cl::NDRange global_work_size(kGlobalSizeM, kGlobalSizeN);
  cl::NDRange local_work_size(32, 32);

  cl::Buffer buffer_A(ocl_kernel.GetKernelContext(),
                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * M * K, (void*) A);
  cl::Buffer buffer_B(ocl_kernel.GetKernelContext(),
                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * K * N, (void*) B);
  cl::Buffer buffer_C(ocl_kernel.GetKernelContext(),
                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * M * N, (void*) C);

  ocl_kernel.set_kernel_args(0, buffer_A, buffer_B, buffer_C, M, N, K, alpha,
                             beta);
  decltype(auto) queue = ocl_kernel.GetCommandQueue();
  decltype(auto) kernel = ocl_kernel.GetKernel();

  cl::Event event;
  ocl_kernel.GetCommandQueue()->enqueueNDRangeKernel(kernel, cl::NullRange,
                                                     global_work_size,
                                                     local_work_size, nullptr,
                                                     &event);
  queue->enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * M * N, C);
  ocl_kernel.ProflingKernel(event);
}

void gemm_v1(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta) {
  const std::string build_options = "-funsafe-max-local-work-size=2";
  OCLKernel ocl_kernel("../src/opencl/gemm/gemm_v1.cl", "gemm_v1_kernel",
                       build_options);
  const int kGlobalSizeM = M;
  const int kGlobalSizeN = N;

  cl::NDRange global_work_size(kGlobalSizeM, kGlobalSizeN);
  cl::NDRange local_work_size(32, 32);

  cl::Buffer buffer_A(ocl_kernel.GetKernelContext(),
                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * M * K, (void*) A);
  cl::Buffer buffer_B(ocl_kernel.GetKernelContext(),
                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * K * N, (void*) B);
  cl::Buffer buffer_C(ocl_kernel.GetKernelContext(),
                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * M * N, (void*) C);

  ocl_kernel.set_kernel_args(0, buffer_A, buffer_B, buffer_C, M, N, K, alpha,
                             beta);
  decltype(auto) queue = ocl_kernel.GetCommandQueue();
  decltype(auto) kernel = ocl_kernel.GetKernel();

  cl::Event event;
  ocl_kernel.GetCommandQueue()->enqueueNDRangeKernel(kernel, cl::NullRange,
                                                     global_work_size,
                                                     local_work_size, nullptr,
                                                     &event);
  queue->enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * M * N, C);
  ocl_kernel.ProflingKernel(event);
}

void gemm_v2(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta) {
  const int kGlobalSizeM = M;
  const int kGlobalSizeN = N;
  constexpr int kBlockSize = 32;

  cl::NDRange global_work_size((kGlobalSizeN + kBlockSize - 1) / kBlockSize *
                                   kBlockSize,
                               (kGlobalSizeM + kBlockSize - 1) / kBlockSize *
                                   kBlockSize);
  cl::NDRange local_work_size(kBlockSize, kBlockSize);

  const std::string build_options =
      " -funsafe-max-local-work-size=2 -expected-thread-mode=SIMD64 "
      "-external-register-sector-mode=2";
  OCLKernel ocl_kernel("../src/opencl/gemm/gemm_v2.cl", "gemm_v2_kernel",
                       build_options);

  cl::Buffer buffer_A(ocl_kernel.GetKernelContext(),
                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * M * K, (void*) A);
  cl::Buffer buffer_B(ocl_kernel.GetKernelContext(),
                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * K * N, (void*) B);
  cl::Buffer buffer_C(ocl_kernel.GetKernelContext(),
                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * M * N, (void*) C);

  ocl_kernel.set_kernel_args(0, buffer_A, buffer_B, buffer_C, M, N, K, alpha,
                             beta);
  decltype(auto) queue = ocl_kernel.GetCommandQueue();
  decltype(auto) kernel = ocl_kernel.GetKernel();

  cl::Event event;
  ocl_kernel.GetCommandQueue()->enqueueNDRangeKernel(kernel, cl::NullRange,
                                                     global_work_size,
                                                     local_work_size, nullptr,
                                                     &event);
  queue->enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * M * N, C);
  ocl_kernel.ProflingKernel(event);
}

void gemm_v3(const float* A, const float* B, float* C, int M, int N, int K,
             float alpha, float beta) {
  const int kGlobalSizeM = M;
  const int kGlobalSizeN = N;
  constexpr int kBlockSize = 32;
  constexpr int kCoarseFactor = 8;

  cl::NDRange global_work_size(kGlobalSizeN / kCoarseFactor,
                               (kGlobalSizeM + kBlockSize - 1) / kBlockSize *
                                   kBlockSize);
  cl::NDRange local_work_size(kBlockSize / kCoarseFactor, kBlockSize);

  const std::string build_options =
      " -funsafe-max-local-work-size=2 -expected-thread-mode=SIMD64 "
      "-external-register-sector-mode=2";

  OCLKernel ocl_kernel("../src/opencl/gemm/gemm_v3.cl", "gemm_v3_kernel",
                       build_options);

  cl::Buffer buffer_A(ocl_kernel.GetKernelContext(),
                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * M * K, (void*) A);
  cl::Buffer buffer_B(ocl_kernel.GetKernelContext(),
                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * K * N, (void*) B);
  cl::Buffer buffer_C(ocl_kernel.GetKernelContext(),
                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * M * N, (void*) C);

  ocl_kernel.set_kernel_args(0, buffer_A, buffer_B, buffer_C, M, N, K, alpha,
                             beta);
  decltype(auto) queue = ocl_kernel.GetCommandQueue();
  decltype(auto) kernel = ocl_kernel.GetKernel();

  cl::Event event;
  ocl_kernel.GetCommandQueue()->enqueueNDRangeKernel(kernel, cl::NullRange,
                                                     global_work_size,
                                                     local_work_size, nullptr,
                                                     &event);
  queue->enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * M * N, C);
  ocl_kernel.ProflingKernel(event);
}
