#include "sgemv.h"
#include "device.h"

void sgemv_v0(const float* A, const float* x, float* y, int M, int N,
              float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel =
      dm.build_kernel("../src/opencl/sgemv/sgemv_v0.cl", "sgemv_v0_kernel");

  cl::NDRange global_work_size(M);
  cl::NDRange local_work_size(256);

  auto ba = dm.create_ro_buffer(sizeof(float) * M * N, A);
  auto bx = dm.create_ro_buffer(sizeof(float) * N, x);
  auto by = dm.create_rw_buffer(sizeof(float) * M, y);

  kernel.setArg(0, ba);
  kernel.setArg(1, bx);
  kernel.setArg(2, by);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, alpha);
  kernel.setArg(6, beta);

  dm.launch(kernel, global_work_size, local_work_size, "sgemv_v0_kernel");
  dm.read_buffer(by, sizeof(float) * M, y);
}

double sgemv_v0_benchmark(const float* A, const float* x, float* y, int M,
                          int N, float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel =
      dm.build_kernel("../src/opencl/sgemv/sgemv_v0.cl", "sgemv_v0_kernel");

  cl::NDRange global_work_size(M);
  cl::NDRange local_work_size(256);

  auto ba = dm.create_ro_buffer(sizeof(float) * M * N, A);
  auto bx = dm.create_ro_buffer(sizeof(float) * N, x);
  auto by = dm.create_rw_buffer(sizeof(float) * M, y);

  kernel.setArg(0, ba);
  kernel.setArg(1, bx);
  kernel.setArg(2, by);
  kernel.setArg(3, M);
  kernel.setArg(4, N);
  kernel.setArg(5, alpha);
  kernel.setArg(6, beta);

  return dm.launch_profiled(kernel, global_work_size, local_work_size,
                            "sgemv_v0_kernel");
}
