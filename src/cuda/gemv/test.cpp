#include "gemv.h"
#include "util.h"
#include <gtest/gtest.h>

TEST(GEMV, gemv_v0) {
  const int m = 512, n = 512;
  std::vector<float> mat_a(m * n, 0.0f);
  std::vector<float> vec_x(n, 0.0f);
  std::vector<float> vec_y(m, 0.0f);
  std::vector<float> cutlass_vec_y(m, 0.0f);
  const float alpha = 1.0f, beta = 0.1f;

  init_random(mat_a, -1.0f, 1.0f);
  init_random(vec_x, -1.0f, 1.0f);
  init_random(vec_y, -1.0f, 1.0f);

  std::copy(vec_y.begin(), vec_y.end(), cutlass_vec_y.begin());

  gemv_v0(mat_a.data(), vec_x.data(), vec_y.data(), m, n, alpha, beta);
  cutlass_gemv_fp32(mat_a.data(), vec_x.data(), cutlass_vec_y.data(), m, n,
                    alpha, beta);

  const float tolerance = 0.0001f;
  for (int i = 0; i < m; ++i) {
    ASSERT_FLOAT_EQ(cutlass_vec_y[i], vec_y[i]);
  }
}

TEST(GEMV, gemv_v1) {
  const int m = 512, n = 512;
  std::vector<float> mat_a(m * n, 0.0f);
  std::vector<float> vec_x(n, 0.0f);
  std::vector<float> vec_y(m, 0.0f);
  std::vector<float> cutlass_vec_y(m, 0.0f);
  const float alpha = 1.0f, beta = 0.5f;

  init_random(mat_a, -1.0f, 1.0f);
  init_random(vec_x, -1.0f, 1.0f);
  init_random(vec_y, -1.0f, 1.0f);

  std::copy(vec_y.begin(), vec_y.end(), cutlass_vec_y.begin());

  gemv_v1(mat_a.data(), vec_x.data(), vec_y.data(), m, n, alpha, beta);
  cutlass_gemv_fp32(mat_a.data(), vec_x.data(), cutlass_vec_y.data(), m, n,
                    alpha, beta);

  const float tolerance = 0.0001f;
  for (int i = 0; i < m; ++i) {
    ASSERT_FLOAT_EQ(cutlass_vec_y[i], vec_y[i]);
  }
}

TEST(GEMV, gemv_v2) {
  const int m = 512, n = 512;
  std::vector<float> mat_a(m * n, 0.0f);
  std::vector<float> vec_x(n, 0.0f);
  std::vector<float> vec_y(m, 0.0f);
  std::vector<float> cutlass_vec_y(m, 0.0f);
  const float alpha = 1.0f, beta = 0.5f;

  init_random(mat_a, -1.0f, 1.0f);
  init_random(vec_x, -1.0f, 1.0f);
  init_random(vec_y, -1.0f, 1.0f);

  std::copy(vec_y.begin(), vec_y.end(), cutlass_vec_y.begin());

  gemv_v2(mat_a.data(), vec_x.data(), vec_y.data(), m, n, alpha, beta);
  cutlass_gemv_fp32(mat_a.data(), vec_x.data(), cutlass_vec_y.data(), m, n,
                    alpha, beta);

  const float tolerance = 0.0001f;
  for (int i = 0; i < m; ++i) {
    EXPECT_LE(std::abs(cutlass_vec_y[i] - vec_y[i]), tolerance);
  }
}
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
  return 0;
}
