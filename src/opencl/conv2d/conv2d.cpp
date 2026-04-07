#include "conv2d.h"
#include <CL/cl.h>
#include <CL/opencl.hpp>
#include "utils.h"

void valid_conv2d_v0(const float* input, const float* filter, float* output,
                     int input_rows, int input_cols, int kernel_rows,
                     int kernel_cols) {
  const std::string build_options = "";
  OCLKernel ocl_kernel("../src/opencl/conv2d/valid_conv2d_v0.cl",
                       "valid_conv2d_v0_kernel", build_options);
  const int kOutputRows = input_rows - kernel_rows + 1;
  const int kOutputCols = input_cols - kernel_cols + 1;

  cl::NDRange global_work_size(kOutputCols, kOutputRows);
  cl::NDRange local_work_size(16, 16);

  cl::Buffer buffer_input(ocl_kernel.GetKernelContext(),
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * input_rows * input_cols,
                          (void*) input);
  cl::Buffer buffer_filter(ocl_kernel.GetKernelContext(),
                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * kernel_rows * kernel_cols,
                           (void*) filter);
  cl::Buffer buffer_output(ocl_kernel.GetKernelContext(),
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * kOutputRows * kOutputCols,
                           (void*) output);

  ocl_kernel.set_kernel_args(0, buffer_input, buffer_filter, buffer_output,
                             input_rows, input_cols, kernel_rows, kernel_cols);
  decltype(auto) queue = ocl_kernel.GetCommandQueue();
  decltype(auto) kernel = ocl_kernel.GetKernel();

  cl::Event event;
  ocl_kernel.GetCommandQueue()->enqueueNDRangeKernel(kernel, cl::NullRange,
                                                     global_work_size,
                                                     cl::NullRange, nullptr,
                                                     &event);
  queue->enqueueReadBuffer(buffer_output, CL_TRUE, 0,
                           sizeof(float) * kOutputRows * kOutputCols, output);
  ocl_kernel.ProflingKernel(event);
}

void valid_conv2d_v1(const float* input, const float* filter, float* output,
                     int input_rows, int input_cols, int kernel_rows,
                     int kernel_cols) {
  const std::string build_options = "";
  OCLKernel ocl_kernel("../src/opencl/conv2d/valid_conv2d_v1.cl",
                       "valid_conv2d_v1_kernel", build_options);
  const int kOutputRows = input_rows - kernel_rows + 1;
  const int kOutputCols = input_cols - kernel_cols + 1;

  cl::NDRange global_work_size(kOutputCols, kOutputRows);
  cl::NDRange local_work_size(16, 16);

  cl::Buffer buffer_input(ocl_kernel.GetKernelContext(),
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * input_rows * input_cols,
                          (void*) input);
  cl::Buffer buffer_filter(ocl_kernel.GetKernelContext(),
                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * kernel_rows * kernel_cols,
                           (void*) filter);
  cl::Buffer buffer_output(ocl_kernel.GetKernelContext(),
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * kOutputRows * kOutputCols,
                           (void*) output);

  ocl_kernel.set_kernel_args(0, buffer_input, buffer_filter, buffer_output,
                             input_rows, input_cols, kernel_rows, kernel_cols);
  decltype(auto) queue = ocl_kernel.GetCommandQueue();
  decltype(auto) kernel = ocl_kernel.GetKernel();

  cl::Event event;
  ocl_kernel.GetCommandQueue()->enqueueNDRangeKernel(kernel, cl::NullRange,
                                                     global_work_size,
                                                     cl::NullRange, nullptr,
                                                     &event);
  queue->enqueueReadBuffer(buffer_output, CL_TRUE, 0,
                           sizeof(float) * kOutputRows * kOutputCols, output);
  ocl_kernel.ProflingKernel(event);
}

void valid_conv2d_v2(const float* input, const float* filter, float* output,
                     int input_rows, int input_cols, int kernel_rows,
                     int kernel_cols) {
  const std::string build_options = "";
  OCLKernel ocl_kernel("../src/opencl/conv2d/valid_conv2d_v2.cl",
                       "valid_conv2d_v2_kernel", build_options);
  const int kOutputRows = input_rows - kernel_rows + 1;
  const int kOutputCols = input_cols - kernel_cols + 1;

  const int kLocalSizeX = 16;
  const int kLocalSizeY = 16;

  const int kSharedInputSizeX = kLocalSizeX + kernel_cols - 1;
  const int kSharedInputSizeY = kLocalSizeY + kernel_rows - 1;

  cl::NDRange global_work_size((kOutputCols + kLocalSizeX - 1) / kLocalSizeX *
                                   kLocalSizeX,
                               (kOutputRows + kLocalSizeY - 1) / kLocalSizeY *
                                   kLocalSizeY);
  cl::NDRange local_work_size(kLocalSizeX, kLocalSizeY);

  cl::Buffer buffer_input(ocl_kernel.GetKernelContext(),
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * input_rows * input_cols,
                          (void*) input);
  cl::Buffer buffer_filter(ocl_kernel.GetKernelContext(),
                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * kernel_rows * kernel_cols,
                           (void*) filter);
  cl::Buffer buffer_output(ocl_kernel.GetKernelContext(),
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * kOutputRows * kOutputCols,
                           (void*) output);
  ocl_kernel.GetKernel().setArg(0, buffer_input);
  ocl_kernel.GetKernel().setArg(1, buffer_filter);
  ocl_kernel.GetKernel().setArg(2, buffer_output);
  ocl_kernel.GetKernel().setArg(
      3, sizeof(float) * (kSharedInputSizeX + 1) * kSharedInputSizeY, nullptr);
  ocl_kernel.GetKernel().setArg(4, input_rows);
  ocl_kernel.GetKernel().setArg(5, input_cols);
  ocl_kernel.GetKernel().setArg(6, kernel_rows);
  ocl_kernel.GetKernel().setArg(7, kernel_cols);
  ocl_kernel.GetKernel().setArg(8, kOutputRows);
  ocl_kernel.GetKernel().setArg(9, kOutputCols);
  decltype(auto) queue = ocl_kernel.GetCommandQueue();
  decltype(auto) kernel = ocl_kernel.GetKernel();

  cl::Event event;
  ocl_kernel.GetCommandQueue()->enqueueNDRangeKernel(kernel, cl::NullRange,
                                                     global_work_size,
                                                     local_work_size, nullptr,
                                                     &event);
  queue->enqueueReadBuffer(buffer_output, CL_TRUE, 0,
                           sizeof(float) * kOutputRows * kOutputCols, output);
  ocl_kernel.ProflingKernel(event);
}

void valid_conv2d_v3(const float* input, const float* filter, float* output,
                     int input_rows, int input_cols, int kernel_rows,
                     int kernel_cols) {
  const std::string build_options = "";
  OCLKernel ocl_kernel("../src/opencl/conv2d/valid_conv2d_v3.cl",
                       "valid_conv2d_v3_kernel", build_options);
  const int kOutputRows = input_rows - kernel_rows + 1;
  const int kOutputCols = input_cols - kernel_cols + 1;
  const int kRegX = 2;
  const int kRegY = 2;
  const int kLocalSizeX = std::min(16, kOutputCols);
  const int kLocalSizeY = std::min(16, kOutputRows);
  const int kGlobalSizeX = (kOutputCols + (kLocalSizeX * kRegX) - 1) /
                           (kLocalSizeX * kRegX) * kLocalSizeX;
  const int kGlobalSizeY = (kOutputRows + (kLocalSizeY * kRegY) - 1) /
                           (kLocalSizeY * kRegY) * kLocalSizeY;

  // std::cout << std::format("global [{}, {}], local [{}, {}]\n", kGlobalSizeY,
  //                          kGlobalSizeX, kLocalSizeY, kLocalSizeX);
  cl::NDRange global_work_size(kGlobalSizeX, kGlobalSizeY);
  cl::NDRange local_work_size(kLocalSizeX, kLocalSizeY);

  cl::Buffer buffer_input(ocl_kernel.GetKernelContext(),
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * input_rows * input_cols,
                          (void*) input);
  cl::Buffer buffer_filter(ocl_kernel.GetKernelContext(),
                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * kernel_rows * kernel_cols,
                           (void*) filter);
  cl::Buffer buffer_output(ocl_kernel.GetKernelContext(),
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(float) * kOutputRows * kOutputCols,
                           (void*) output);

  ocl_kernel.set_kernel_args(0, buffer_input, buffer_filter, buffer_output,
                             input_rows, input_cols, kernel_rows, kernel_cols,
                             kOutputRows, kOutputCols);
  decltype(auto) queue = ocl_kernel.GetCommandQueue();
  decltype(auto) kernel = ocl_kernel.GetKernel();

  cl::Event event;
  ocl_kernel.GetCommandQueue()->enqueueNDRangeKernel(kernel, cl::NullRange,
                                                     global_work_size,
                                                     local_work_size, nullptr,
                                                     &event);
  queue->enqueueReadBuffer(buffer_output, CL_TRUE, 0,
                           sizeof(float) * kOutputRows * kOutputCols, output);
  ocl_kernel.ProflingKernel(event);
}