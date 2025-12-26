#include "gemm.h"
#include <CL/cl.h>
#include <CL/cl2.hpp>
#include <CL/opencl.hpp>
#include <vector>
#include "utils.h"

void gemm_v0(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
  const std::string build_options = "";

  cl::Context context = cl::Context::getDefault();
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  cl::Device device = devices[0];

  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

  const std::string kernel_src = read_file("../src/opencl/gemm/gemm_v0.cl");
  cl::Program program(context, kernel_src);
  program.build({device}, build_options.c_str());
  cl::Kernel kernel(program, "gemm_v0_kernel");

  const int kGlobalSizeM = M;
  const int kGlobalSizeN = N;

  cl::NDRange global_work_size(kGlobalSizeM, kGlobalSizeN);

  cl::Buffer buffer_A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * M * K, (void*)A);
  cl::Buffer buffer_B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * K * N, (void*)B);
  cl::Buffer buffer_C(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * M * N, (void*)C);

  kernel.setArg(0, buffer_A);
  kernel.setArg(1, buffer_B);
  kernel.setArg(2, buffer_C);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, K);
  kernel.setArg(6, alpha);
  kernel.setArg(7, beta);

  cl::Event event;
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, cl::NullRange, nullptr, &event);
  queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * M * N, C);
  print_kernel_profiling_info("gemm_v0_kernel", event);
}

void gemm_v1(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
  const std::string build_options = "";

  cl::Context context = cl::Context::getDefault();
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  cl::Device device = devices[0];

  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

  const std::string kernel_src = read_file("../src/opencl/gemm/gemm_v1.cl");
  cl::Program program(context, kernel_src);
  program.build({device}, build_options.c_str());
  cl::Kernel kernel(program, "gemm_v1_kernel");

  const int kGlobalSizeM = M;
  const int kGlobalSizeN = N;

  cl::NDRange global_work_size(kGlobalSizeM, kGlobalSizeN);

  cl::Buffer buffer_A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * M * K, (void*)A);
  cl::Buffer buffer_B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * K * N, (void*)B);
  cl::Buffer buffer_C(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * M * N, (void*)C);

  kernel.setArg(0, buffer_A);
  kernel.setArg(1, buffer_B);
  kernel.setArg(2, buffer_C);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, K);
  kernel.setArg(6, alpha);
  kernel.setArg(7, beta);

  cl::Event event;
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, cl::NullRange, nullptr, &event);
  queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * M * N, C);
  print_kernel_profiling_info("gemm_v1_kernel", event);
}