#include "utils.h"
#include <CL/opencl.hpp>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>
#include "reduce.h"

float reduce_ref(const std::vector<float>& input) {
  return std::accumulate(input.begin(), input.end(), 0.0f);
}

TEST(ReduceTest, reduce_v0) {
  const int n = 2048;
  std::vector<float> input(n, 0.0f);
  set_random_values(input, -1.0f, 1.0f);

  float cpu_output = reduce_ref(input);
  float ocl_output = 0.0f;
  reduce_v0(input.data(), &ocl_output, n);

  constexpr float kEpsilon = 1e-3f;
  EXPECT_NEAR(ocl_output, cpu_output, kEpsilon);
}

TEST(ReduceTest, reduce_v1) {
  const int n = 2048;
  std::vector<float> input(n, 0.0f);
  set_random_values(input, -1.0f, 1.0f);

  float cpu_output = reduce_ref(input);
  float ocl_output = 0.0f;
  reduce_v1(input.data(), &ocl_output, n);

  constexpr float kEpsilon = 1e-3f;
  EXPECT_NEAR(ocl_output, cpu_output, kEpsilon);
}

TEST(ReduceTest, reduce_v2) {
  const int n = 2048;
  std::vector<float> input(n, 0.0f);
  set_random_values(input, -1.0f, 1.0f);

  float cpu_output = reduce_ref(input);
  float ocl_output = 0.0f;
  reduce_v2(input.data(), &ocl_output, n);

  constexpr float kEpsilon = 1e-3f;
  EXPECT_NEAR(ocl_output, cpu_output, kEpsilon);
}

TEST(ReduceTest, reduce_v3) {
  const int n = 2048;
  std::vector<float> input(n, 0.0f);
  set_random_values(input, -1.0f, 1.0f);

  float cpu_output = reduce_ref(input);
  float ocl_output = 0.0f;
  reduce_v3(input.data(), &ocl_output, n);

  constexpr float kEpsilon = 1e-3f;
  EXPECT_NEAR(ocl_output, cpu_output, kEpsilon);
}

TEST(ReduceTest, reduce_v4) {
  const int n = 2048;
  std::vector<float> input(n, 0.0f);
  set_random_values(input, 1.0f, 1.0f);

  float cpu_output = reduce_ref(input);
  float ocl_output = 0.0f;
  reduce_v4(input.data(), &ocl_output, n);

  constexpr float kEpsilon = 1e-3f;
  EXPECT_NEAR(ocl_output, cpu_output, kEpsilon);
}