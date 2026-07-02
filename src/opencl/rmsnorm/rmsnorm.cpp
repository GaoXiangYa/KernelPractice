#include "rmsnorm.h"
#include <CL/opencl.hpp>
#include "device.h"
// ---------------------------------------------------------------------------
// v0 — local memory tree reduce
// ---------------------------------------------------------------------------
void launch_rmsnorm_v0(const float* input, const float* weight, float* output,
                       int N, int D, float epsilon) {
  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/rmsnorm/rmsnorm_v0.cl",
                                "rms_norm_fused_v0");

  size_t elems = (size_t) N * D;

  auto d_input = dm.create_ro_buffer(sizeof(float) * elems, input);
  auto d_weight = dm.create_ro_buffer(sizeof(float) * D, weight);
  auto d_output = dm.create_rw_buffer(sizeof(float) * elems, output);

  int lsz = 64;

  kernel.setArg(0, d_input);
  kernel.setArg(1, d_weight);
  kernel.setArg(2, d_output);
  kernel.setArg(3, D);
  kernel.setArg(4, epsilon);
  kernel.setArg(5, cl::Local(lsz * sizeof(float)));

  cl::NDRange global(lsz * N);
  cl::NDRange local(lsz);

  dm.launch(kernel, global, local, "rms_norm_fused_v0");
  dm.read_buffer(d_output, sizeof(float) * elems, output);
}

void launch_rmsnorm_v0(const std::vector<float>& input,
                       const std::vector<float>& weight,
                       std::vector<float>& output, int N, int D,
                       float epsilon) {
  launch_rmsnorm_v0(input.data(), weight.data(), output.data(), N, D, epsilon);
}

// ---------------------------------------------------------------------------
// v1 — float4 + sub_group_reduce  (caller configures global / local)
// ---------------------------------------------------------------------------
void launch_rmsnorm_v1(const float* input, const float* weight, float* output,
                       int N, int D, float epsilon) {
  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/rmsnorm/rmsnorm_v1.cl",
                                "rms_norm_fused_v1");

  size_t elems = (size_t) N * D;

  auto d_input = dm.create_ro_buffer(sizeof(float) * elems, input);
  auto d_weight = dm.create_ro_buffer(sizeof(float) * D, weight);
  auto d_output = dm.create_rw_buffer(sizeof(float) * elems, output);

  kernel.setArg(0, d_input);
  kernel.setArg(1, d_weight);
  kernel.setArg(2, d_output);
  kernel.setArg(3, D);
  kernel.setArg(4, epsilon);

  int lsz = 64;

  cl::NDRange global(lsz * N);
  cl::NDRange local(lsz);

  dm.launch(kernel, global, local, "rms_norm_fused_v1");
  dm.read_buffer(d_output, sizeof(float) * elems, output);
}

void launch_rmsnorm_v1(const std::vector<float>& input,
                       const std::vector<float>& weight,
                       std::vector<float>& output, int N, int D,
                       float epsilon) {
  launch_rmsnorm_v1(input.data(), weight.data(), output.data(), N, D, epsilon);
}

void launch_rmsnorm_v2(const float* input, const float* weight, float* output,
                       int N, int D, float epsilon) {
  auto& dm = DeviceManager::get();
  auto kernel = dm.build_kernel("../src/opencl/rmsnorm/rmsnorm_v2.cl",
                                "rms_norm_fused_v2");

  size_t elems = (size_t) N * D;
  const int max_sub_group_size =
      kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
          cl::Device::getDefault());
  const int max_work_group_size =
      cl::Device::getDefault().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  const int tmp_size =
      (max_work_group_size + max_sub_group_size - 1) / max_sub_group_size;

  std::cout << max_sub_group_size << " " << max_work_group_size << " "
            << tmp_size << std::endl;
  auto d_input = dm.create_ro_buffer(sizeof(float) * elems, input);
  auto d_weight = dm.create_ro_buffer(sizeof(float) * D, weight);
  auto d_output = dm.create_rw_buffer(sizeof(float) * elems, output);
  auto d_tmp = dm.create_rw_buffer(sizeof(float) * tmp_size, nullptr);

  kernel.setArg(0, d_input);
  kernel.setArg(1, d_weight);
  kernel.setArg(2, d_output);
  kernel.setArg(3, D);
  kernel.setArg(4, epsilon);
  kernel.setArg(5, d_tmp);

  const int lsz = 256;
  cl::NDRange global(lsz * N);
  cl::NDRange local(lsz);

  dm.launch(kernel, global, local, "rms_norm_fused_v2");
  dm.read_buffer(d_output, sizeof(float) * elems, output);
}

void launch_rmsnorm_v2(const std::vector<float>& input,
                       const std::vector<float>& weight,
                       std::vector<float>& output, int N, int D,
                       float epsilon) {
  launch_rmsnorm_v2(input.data(), weight.data(), output.data(), N, D, epsilon);
}