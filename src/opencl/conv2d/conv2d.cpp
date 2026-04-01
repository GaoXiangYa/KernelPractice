#include "conv2d.h"
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