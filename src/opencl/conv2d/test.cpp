#include <gtest/gtest.h>
#include "conv2d.h"
#include "utils.h"

void conv2d_ref(const float* input, const float* kernel, float* output,
                int input_rows, int input_cols, int kernel_rows,
                int kernel_cols) {
  int out_rows = input_rows - kernel_rows + 1;
  int out_cols = input_cols - kernel_cols + 1;

  for (int i = 0; i < out_rows; ++i) {
    for (int j = 0; j < out_cols; ++j) {
      float sum = 0.0f;

      // 遍历卷积核
      for (int r = 0; r < kernel_rows; ++r) {
        for (int c = 0; c < kernel_cols; ++c) {
          int input_r = i + r;
          int input_c = j + c;

          float val = input[input_r * input_cols + input_c];
          float w = kernel[r * kernel_cols + c];

          sum += val * w;
        }
      }

      output[i * out_cols + j] = sum;
    }
  }
}

TEST(CONV2D, valid_conv2d_v0) {
  const int input_rows = 3072;
  const int input_cols = 3072;
  const int kernel_rows = 15;
  const int kernel_cols = 15;
  const int output_rows = input_rows - kernel_rows + 1;
  const int output_cols = input_cols - kernel_cols + 1;

  std::vector<float> input(input_rows * input_cols, 0.0f);
  std::vector<float> filter(kernel_rows * kernel_cols, 0.0f);
  std::vector<float> output_cpu(output_rows * output_cols, 0.0f);
  std::vector<float> output_ocl(output_rows * output_cols, 0.0f);

  set_random_values(input, -1.0f, 1.0f);
  set_random_values(filter, -1.0f, 1.0f);

  conv2d_ref(input.data(), filter.data(), output_cpu.data(), input_rows,
             input_cols, kernel_rows, kernel_cols);
  valid_conv2d_v0(input.data(), filter.data(), output_ocl.data(), input_rows,
                  input_cols, kernel_rows, kernel_cols);

  constexpr float kEpsilon = 1e-3f;
  for (int i = 0; i < output_rows * output_cols; ++i) {
    ASSERT_NEAR(output_ocl[i], output_cpu[i], kEpsilon);
  }
}

TEST(CONV2D, valid_conv2d_v1) {
  const int input_rows = 3072;
  const int input_cols = 3072;
  const int kernel_rows = 15;
  const int kernel_cols = 15;
  const int output_rows = input_rows - kernel_rows + 1;
  const int output_cols = input_cols - kernel_cols + 1;

  std::vector<float> input(input_rows * input_cols, 0.0f);
  std::vector<float> filter(kernel_rows * kernel_cols, 0.0f);
  std::vector<float> output_cpu(output_rows * output_cols, 0.0f);
  std::vector<float> output_ocl(output_rows * output_cols, 0.0f);

  set_random_values(input, -1.0f, 1.0f);
  set_random_values(filter, -1.0f, 1.0f);

  conv2d_ref(input.data(), filter.data(), output_cpu.data(), input_rows,
             input_cols, kernel_rows, kernel_cols);
  valid_conv2d_v1(input.data(), filter.data(), output_ocl.data(), input_rows,
                  input_cols, kernel_rows, kernel_cols);

  constexpr float kEpsilon = 1e-3f;
  for (int i = 0; i < output_rows * output_cols; ++i) {
    ASSERT_NEAR(output_ocl[i], output_cpu[i], kEpsilon);
  }
}

TEST(CONV2D, valid_conv2d_v2) {
  const int input_rows = 3072;
  const int input_cols = 3072;
  const int kernel_rows = 15;
  const int kernel_cols = 15;
  const int output_rows = input_rows - kernel_rows + 1;
  const int output_cols = input_cols - kernel_cols + 1;

  std::vector<float> input(input_rows * input_cols, 0.0f);
  std::vector<float> filter(kernel_rows * kernel_cols, 0.0f);
  std::vector<float> output_cpu(output_rows * output_cols, 0.0f);
  std::vector<float> output_ocl(output_rows * output_cols, 0.0f);

  set_random_values(input, -1.0f, 1.0f);
  set_random_values(filter, -1.0f, 1.0f);

  conv2d_ref(input.data(), filter.data(), output_cpu.data(), input_rows,
             input_cols, kernel_rows, kernel_cols);
  valid_conv2d_v2(input.data(), filter.data(), output_ocl.data(), input_rows,
                  input_cols, kernel_rows, kernel_cols);

  constexpr float kEpsilon = 1e-3f;
  for (int i = 0; i < output_rows * output_cols; ++i) {
    ASSERT_NEAR(output_ocl[i], output_cpu[i], kEpsilon);
  }
}

TEST(CONV2D, valid_conv2d_v3) {
  const int input_rows = 3072;
  const int input_cols = 3072;
  const int kernel_rows = 15;
  const int kernel_cols = 15;
  const int output_rows = input_rows - kernel_rows + 1;
  const int output_cols = input_cols - kernel_cols + 1;

  std::vector<float> input(input_rows * input_cols, 0.0f);
  std::vector<float> filter(kernel_rows * kernel_cols, 0.0f);
  std::vector<float> output_cpu(output_rows * output_cols, 0.0f);
  std::vector<float> output_ocl(output_rows * output_cols, 0.0f);

  set_random_values(input, -1.0f, 1.0f);
  set_random_values(filter, -1.0f, 1.0f);

  conv2d_ref(input.data(), filter.data(), output_cpu.data(), input_rows,
             input_cols, kernel_rows, kernel_cols);
  valid_conv2d_v3(input.data(), filter.data(), output_ocl.data(), input_rows,
                  input_cols, kernel_rows, kernel_cols);

  constexpr float kEpsilon = 1e-3f;
  for (int i = 0; i < output_rows * output_cols; ++i) {
    ASSERT_NEAR(output_ocl[i], output_cpu[i], kEpsilon);
  }
}
