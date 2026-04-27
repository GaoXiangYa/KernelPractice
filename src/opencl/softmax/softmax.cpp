#include "softmax.h"
#include "utils.h"

void launchSoftmaxV0(float* h_output, const float* h_input, const float* h_mask,
                     int batch_size, int num_heads, int seq_q, int seq_k,
                     float scale, int is_causal) {
  // Placeholder for the actual OpenCL kernel launch code.
  // This function should set up the OpenCL environment, compile the kernel,
  // and execute it with the provided parameters.
  const std::string build_options = "";
  OCLKernel ocl_kernel("../src/opencl/softmax/softmax_v0.cl", "softmaxKernelV0",
                       build_options);
  const int kThreadNum = 64;
  cl::NDRange global_work_size(seq_q * kThreadNum, num_heads, batch_size);
  cl::NDRange local_work_size(kThreadNum, 1, 1);

  cl::Buffer d_output(ocl_kernel.GetKernelContext(),
                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * seq_q * seq_k * batch_size * num_heads,
                      (void*) h_output);

  cl::Buffer d_input(ocl_kernel.GetKernelContext(),
                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * seq_q * seq_k * batch_size * num_heads,
                     (void*) h_input);

  cl::Buffer d_mask(ocl_kernel.GetKernelContext(),
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    sizeof(float) * seq_q * seq_k * batch_size * num_heads,
                    (void*) h_mask);

  ocl_kernel.set_kernel_args(0, d_output, d_input, d_mask, batch_size,
                             num_heads, seq_q, seq_k, scale, is_causal);
  decltype(auto) queue = ocl_kernel.GetCommandQueue();
  decltype(auto) kernel = ocl_kernel.GetKernel();

  cl::Event event;
  ocl_kernel.GetCommandQueue()->enqueueNDRangeKernel(kernel, cl::NullRange,
                                                     global_work_size,
                                                     local_work_size, nullptr,
                                                     &event);
  queue->enqueueReadBuffer(d_output, CL_TRUE, 0,
                           sizeof(float) * seq_q * seq_k * batch_size *
                               num_heads,
                           h_output);
  ocl_kernel.ProflingKernel(event);
}