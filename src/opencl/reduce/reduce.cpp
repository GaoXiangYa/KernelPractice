#include "reduce.h"
#include "../utils/utils.h"
#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>

void reduce_v0(const float *input, float *output, int n) {
  const std::string build_options = "";

  cl::Context context = cl::Context::getDefault();
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  cl::Device device = devices[0];

  cl::CommandQueue queue(context, device);

  const std::string kernel_src = read_file(
      "../src/opencl/reduce/reduce_v0.cl");
  cl::Program program(context, kernel_src);
  program.build({device}, build_options.c_str());
  cl::Kernel kernel(program, "reduce_v0_kernel");

  
  constexpr int kThreadsPerBlock = 256;
  const int kNumBlocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock * kThreadsPerBlock;

  std::vector<float> temp_output(kNumBlocks, 0.0f);

  cl::Buffer input_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * n, (void *)input);
  cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * kNumBlocks,
                           (void *)temp_output.data());

  kernel.setArg(0, input_buffer);
  kernel.setArg(1, output_buffer);
  kernel.setArg(2, n);

  cl::NDRange global_work_size(kNumBlocks);
  cl::NDRange local_work_size(kThreadsPerBlock);
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size,
                             local_work_size);
  queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(float) * kNumBlocks, temp_output.data());

  for (int i = 0; i < kNumBlocks; ++i) {
    *output += temp_output[i];
  }
}