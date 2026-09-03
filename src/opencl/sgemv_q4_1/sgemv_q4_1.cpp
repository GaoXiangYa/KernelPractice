#include "sgemv_q4_1.h"
#include "device.h"

void sgemv_q4_1_v0(const unsigned char* A_q4, const float* x, float* y, int M,
                   int N, int block_k, float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/sgemv_q4_1/sgemv_q4_1_v0.cl",
                                "sgemv_q4_1_v0_kernel");

  const int blocks_per_row = N / block_k;
  const int block_bytes = (int) (sizeof(float) * 2 + block_k / 2);

  cl::NDRange global_work_size(M);
  cl::NDRange local_work_size(256);

  auto ba = dm.create_ro_buffer((size_t) M * blocks_per_row * block_bytes,
                                A_q4);
  auto bx = dm.create_ro_buffer(sizeof(float) * N, x);
  auto by = dm.create_rw_buffer(sizeof(float) * M, y);

  kernel.setArg(0, ba);
  kernel.setArg(1, bx);
  kernel.setArg(2, by);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, block_k);
  kernel.setArg(6, alpha);
  kernel.setArg(7, beta);

  dm.launch(kernel, global_work_size, local_work_size,
            "sgemv_q4_1_v0_kernel");
  dm.read_buffer(by, sizeof(float) * M, y);
}

double sgemv_q4_1_v0_benchmark(const unsigned char* A_q4, const float* x,
                               float* y, int M, int N, int block_k,
                               float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/sgemv_q4_1/sgemv_q4_1_v0.cl",
                                "sgemv_q4_1_v0_kernel");

  const int blocks_per_row = N / block_k;
  const int block_bytes = (int) (sizeof(float) * 2 + block_k / 2);

  cl::NDRange global_work_size(M);
  cl::NDRange local_work_size(256);

  auto ba = dm.create_ro_buffer((size_t) M * blocks_per_row * block_bytes,
                                A_q4);
  auto bx = dm.create_ro_buffer(sizeof(float) * N, x);
  auto by = dm.create_rw_buffer(sizeof(float) * M, y);

  kernel.setArg(0, ba);
  kernel.setArg(1, bx);
  kernel.setArg(2, by);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, block_k);
  kernel.setArg(6, alpha);
  kernel.setArg(7, beta);

  return dm.launch_profiled(kernel, global_work_size, local_work_size,
                            "sgemv_q4_1_v0_kernel");
}
