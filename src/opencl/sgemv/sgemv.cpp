#include "sgemv.h"
#include <CL/cl.h>
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

  constexpr int kThreadX = 256;
  const int kGlobalX = (M + kThreadX - 1) / kThreadX * kThreadX;
  cl::NDRange global_work_size(kGlobalX);
  cl::NDRange local_work_size(kThreadX);

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

void sgemv_v1(const float* A, const float* x, float* y, int M, int N,
              float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel =
      dm.build_kernel("../src/opencl/sgemv/sgemv_v1.cl", "sgemv_v1_kernel");

  constexpr int kThreadX = 256;
  cl::NDRange local_work_size(kThreadX);

  const int kSubGroupSize = dm.get_subgroup_size(kernel, local_work_size);
  const int kRowsPerGroup = kThreadX / kSubGroupSize;
  const int kNumGroups = (M + kRowsPerGroup - 1) / kRowsPerGroup;
  cl::NDRange global_work_size(kNumGroups * kThreadX);

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

  dm.launch(kernel, global_work_size, local_work_size, "sgemv_v1_kernel");
  dm.read_buffer(by, sizeof(float) * M, y);
}

double sgemv_v1_benchmark(const float* A, const float* x, float* y, int M,
                          int N, float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel =
      dm.build_kernel("../src/opencl/sgemv/sgemv_v1.cl", "sgemv_v1_kernel");

  constexpr int kThreadX = 256;
  cl::NDRange local_work_size(kThreadX);

  const int kSubGroupSize = dm.get_subgroup_size(kernel, local_work_size);
  const int kRowsPerGroup = kThreadX / kSubGroupSize;
  const int kNumGroups = (M + kRowsPerGroup - 1) / kRowsPerGroup;
  cl::NDRange global_work_size(kNumGroups * kThreadX);

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
                            "sgemv_v1_kernel");
}

void sgemv_v2(const float* A, const float* x, float* y, int M, int N,
              float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel =
      dm.build_kernel("../src/opencl/sgemv/sgemv_v2.cl", "sgemv_v2_kernel");

  constexpr int kThreadX = 256;
  cl::NDRange local_work_size(kThreadX);

  const int kSubGroupSize = dm.get_subgroup_size(kernel, local_work_size);
  const int kRowsPerGroup = kThreadX / kSubGroupSize;
  const int kNumGroups = (M + kRowsPerGroup - 1) / kRowsPerGroup;
  cl::NDRange global_work_size(kNumGroups * kThreadX);

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

  dm.launch(kernel, global_work_size, local_work_size, "sgemv_v2_kernel");
  dm.read_buffer(by, sizeof(float) * M, y);
}

double sgemv_v2_benchmark(const float* A, const float* x, float* y, int M,
                          int N, float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel =
      dm.build_kernel("../src/opencl/sgemv/sgemv_v2.cl", "sgemv_v2_kernel");

  constexpr int kThreadX = 256;
  cl::NDRange local_work_size(kThreadX);

  const int kSubGroupSize = dm.get_subgroup_size(kernel, local_work_size);
  const int kRowsPerGroup = kThreadX / kSubGroupSize;
  const int kNumGroups = (M + kRowsPerGroup - 1) / kRowsPerGroup;
  cl::NDRange global_work_size(kNumGroups * kThreadX);

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
                            "sgemv_v2_kernel");
}

void sgemv_v3(const float* A, const float* x, float* y, int M, int N,
              float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel =
      dm.build_kernel("../src/opencl/sgemv/sgemv_v3.cl", "sgemv_v3_kernel");

  constexpr int kThreadX = 256;
  cl::NDRange local_work_size(kThreadX);

  const int kSubGroupSize = dm.get_subgroup_size(kernel, local_work_size);
  const int kRowsPerGroup = kThreadX / kSubGroupSize;
  const int kNumGroups = (M + kRowsPerGroup - 1) / kRowsPerGroup;
  cl::NDRange global_work_size(kNumGroups * kThreadX);

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

  dm.launch(kernel, global_work_size, local_work_size, "sgemv_v3_kernel");
  dm.read_buffer(by, sizeof(float) * M, y);
}

double sgemv_v3_benchmark(const float* A, const float* x, float* y, int M,
                          int N, float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel =
      dm.build_kernel("../src/opencl/sgemv/sgemv_v3.cl", "sgemv_v3_kernel");

  constexpr int kThreadX = 256;
  cl::NDRange local_work_size(kThreadX);

  const int kSubGroupSize = dm.get_subgroup_size(kernel, local_work_size);
  const int kRowsPerGroup = kThreadX / kSubGroupSize;
  const int kNumGroups = (M + kRowsPerGroup - 1) / kRowsPerGroup;
  cl::NDRange global_work_size(kNumGroups * kThreadX);

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
                            "sgemv_v3_kernel");
}

void sgemv_v4(const float* A, const float* x, float* y, int M, int N,
              float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel =
      dm.build_kernel("../src/opencl/sgemv/sgemv_v4.cl", "sgemv_v4_kernel");

  constexpr int kThreadX = 256;
  cl::NDRange local_work_size(kThreadX);

  const int kSubGroupSize = dm.get_subgroup_size(kernel, local_work_size);
  const int kRowsPerGroup = kThreadX / kSubGroupSize;
  const int kNumGroups = (M + kRowsPerGroup - 1) / kRowsPerGroup;
  cl::NDRange global_work_size(kNumGroups * kThreadX);

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

  dm.launch(kernel, global_work_size, local_work_size, "sgemv_v4_kernel");
  dm.read_buffer(by, sizeof(float) * M, y);
}

double sgemv_v4_benchmark(const float* A, const float* x, float* y, int M,
                          int N, float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel =
      dm.build_kernel("../src/opencl/sgemv/sgemv_v4.cl", "sgemv_v4_kernel");

  constexpr int kThreadX = 256;
  cl::NDRange local_work_size(kThreadX);

  const int kSubGroupSize = dm.get_subgroup_size(kernel, local_work_size);
  const int kRowsPerGroup = kThreadX / kSubGroupSize;
  const int kNumGroups = (M + kRowsPerGroup - 1) / kRowsPerGroup;
  cl::NDRange global_work_size(kNumGroups * kThreadX);

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
                            "sgemv_v4_kernel");
}

void sgemv_v5(const float* A, const float* x, float* y, int M, int N,
              float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel =
      dm.build_kernel("../src/opencl/sgemv/sgemv_v5.cl", "sgemv_v5_kernel");

  constexpr int kThreadX = 256;
  cl::NDRange local_work_size(kThreadX);

  const int kSubGroupSize = dm.get_subgroup_size(kernel, local_work_size);
  const int kRowsPerGroup = kThreadX / kSubGroupSize * 4;
  const int kNumGroups = (M + kRowsPerGroup - 1) / kRowsPerGroup;
  cl::NDRange global_work_size(kNumGroups * kThreadX);

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

  dm.launch(kernel, global_work_size, local_work_size, "sgemv_v4_kernel");
  dm.read_buffer(by, sizeof(float) * M, y);
}

double sgemv_v5_benchmark(const float* A, const float* x, float* y, int M,
                          int N, float alpha, float beta) {
  auto& dm = DeviceManager::get();
  auto kernel =
      dm.build_kernel("../src/opencl/sgemv/sgemv_v5.cl", "sgemv_v5_kernel");

  constexpr int kThreadX = 256;
  cl::NDRange local_work_size(kThreadX);

  const int kSubGroupSize = dm.get_subgroup_size(kernel, local_work_size);
  const int kRowsPerGroup = kThreadX / kSubGroupSize * 4;
  const int kNumGroups = (M + kRowsPerGroup - 1) / kRowsPerGroup;
  cl::NDRange global_work_size(kNumGroups * kThreadX);

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
                            "sgemv_v5_kernel");
}