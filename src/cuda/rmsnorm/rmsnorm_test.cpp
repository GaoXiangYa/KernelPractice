#include "rmsnorm.h"
#include "util.h"
#include <gtest/gtest.h>
#include <vector>

TEST(rmsnorm, rmsnorm_v0) {
  const int len = 8192;
  std::vector<float> input(len, 0.0f);
  std::vector<float> output(len, -1.0f);
  std::vector<float> flashinfer_output(len, 0.0f);
  std::vector<float> weight(len, 1.0f);
  const float eps = 1.0000f;

  init_random(input, 0.0f, 1.0f);

  flashinfer_rmsnorm(input.data(), weight.data(), flashinfer_output.data(), len,
                     eps);

  rmsnorm_v0(input.data(), output.data(), len, eps);

  const float tolerance = 0.01f;
  for (int i = 0; i < len; ++i) {
    // std::cout << flashinfer_output[i] << " " << output[i] << "\n";
    EXPECT_LE(std::abs(flashinfer_output[i] - output[i]), tolerance);
  }
}

TEST(rmsnorm, rmsnorm_v1) {
  const int len = 8192;
  std::vector<float> input(len, 0.0f);
  std::vector<float> output(len, -1.0f);
  std::vector<float> flashinfer_output(len, 0.0f);
  std::vector<float> weight(len, 1.0f);
  const float eps = 1.0000f;

  init_random(input, -2.0f, 2.0f);

  flashinfer_rmsnorm(input.data(), weight.data(), flashinfer_output.data(), len,
                     eps);

  rmsnorm_v1(input.data(), output.data(), len, eps);

  const float tolerance = 0.1f;
  for (int i = 0; i < len; ++i) {
    EXPECT_LE(std::abs(flashinfer_output[i] - output[i]), tolerance);
  }
}

TEST(rmsnorm, rmsnorm_v2) {
  const int len = 8192;
  std::vector<float> input(len, 0.0f);
  std::vector<float> output(len, -1.0f);
  std::vector<float> flashinfer_output(len, 0.0f);
  std::vector<float> weight(len, 1.0f);
  const float eps = 1.0000f;

  init_random(input, -2.0f, 2.0f);

  flashinfer_rmsnorm(input.data(), weight.data(), flashinfer_output.data(), len,
                     eps);

  rmsnorm_v2(input.data(), output.data(), len, eps);

  const float tolerance = 0.1f;
  for (int i = 0; i < len; ++i) {
    EXPECT_LE(std::abs(flashinfer_output[i] - output[i]), tolerance);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
  return 0;
}