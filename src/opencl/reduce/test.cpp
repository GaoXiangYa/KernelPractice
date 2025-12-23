#include <CL/opencl.hpp>
#include <gtest/gtest.h>
#include "../utils/utils.h"

TEST(ReduceTest, reduce_v0) {
  // Expect two strings to be equal.
  cl::Context context = cl::Context::getDefault();
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  cl::Device device = devices[0];
  print_opencl_limits(device.get(), nullptr);
}