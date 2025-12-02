#include "gemv.h"
#include "util.h"
#include <gtest/gtest.h>
#include <format>
#include <iostream>

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

TEST(GEMV, gemv_v3) {
  const int m = 1024, n = 1024;
  std::vector<float> mat_a(m * n, 0.0f);
  std::vector<float> vec_x(n, 0.0f);
  std::vector<float> vec_y(m, 0.0f);
  std::vector<float> cutlass_vec_y(m, 0.0f);
  const float alpha = 1.0f, beta = 0.5f;

  init_random(mat_a, -1.0f, 1.0f);
  init_random(vec_x,-1.0f, 1.0f);
  init_random(vec_y,-1.0f, 1.0f);

  std::copy(vec_y.begin(), vec_y.end(), cutlass_vec_y.begin());

  gemv_v3(mat_a.data(), vec_x.data(), vec_y.data(), m, n, alpha, beta);
  cutlass_gemv_fp32(mat_a.data(), vec_x.data(), cutlass_vec_y.data(), m, n,
                    alpha, beta);

  const float tolerance = 0.0001f;
  for (int i = 0; i < m; ++i) {
    // std::cout << cutlass_vec_y[i] << " " << vec_y[i] << "\n";
    if (std::abs(cutlass_vec_y[i] - vec_y[i]) >= tolerance) {
      std::cout << std::format("cutlass {}, gemv_v3 {}\n", cutlass_vec_y[i],
                               vec_y[i]);
      ASSERT_TRUE(false);
    }
  }
}

TEST(GEMV, gemv_v4) {
  const int m = 1024, n = 1024;
  std::vector<float> mat_a(m * n, 0.0f);
  std::vector<float> vec_x(n, 0.0f);
  std::vector<float> vec_y(m, 0.0f);
  std::vector<float> cutlass_vec_y(m, 0.0f);
  const float alpha = 1.0f, beta = -0.01f;

  init_random(mat_a, -1.0f, 1.0f);
  init_random(vec_x,-1.0f, 1.0f);
  init_random(vec_y,-1.0f, 1.0f);

  std::copy(vec_y.begin(), vec_y.end(), cutlass_vec_y.begin());

  gemv_v4(mat_a.data(), vec_x.data(), vec_y.data(), m, n, alpha, beta);
  cutlass_gemv_fp32(mat_a.data(), vec_x.data(), cutlass_vec_y.data(), m, n,
                    alpha, beta);

  const float tolerance = 0.01f;
  for (int i = 0; i < m; ++i) {
    // std::cout << cutlass_vec_y[i] << " " << vec_y[i] << "\n";
    if (std::abs(cutlass_vec_y[i] - vec_y[i]) >= tolerance) {
      std::cout << std::format("cutlass {}, gemv_v4 {}\n", cutlass_vec_y[i],
                               vec_y[i]);
      ASSERT_TRUE(false);
    }
  }
}

TEST(GEMV, gemv_v5) {
  const int m = 1024, n = 1024;
  std::vector<float> mat_a(m * n, 0.0f);
  std::vector<float> vec_x(n, 0.0f);
  std::vector<float> vec_y(m, 0.0f);
  std::vector<float> cutlass_vec_y(m, 0.0f);
  const float alpha = 1.0f, beta = 0.01f;

  init_random(mat_a, -1.0f, 1.0f);
  init_random(vec_x,-1.0f, 1.0f);
  init_random(vec_y,-1.0f, 1.0f);

  std::copy(vec_y.begin(), vec_y.end(), cutlass_vec_y.begin());

  gemv_v5(mat_a.data(), vec_x.data(), vec_y.data(), m, n, alpha, beta);
  cutlass_gemv_fp32(mat_a.data(), vec_x.data(), cutlass_vec_y.data(), m, n,
                    alpha, beta);

  const float tolerance = 0.01f;
  for (int i = 0; i < m; ++i) {
    // std::cout << cutlass_vec_y[i] << " " << vec_y[i] << "\n";
    if (std::abs(cutlass_vec_y[i] - vec_y[i]) >= tolerance) {
      std::cout << std::format("cutlass {}, gemv_v5 {}\n", cutlass_vec_y[i],
                               vec_y[i]);
      ASSERT_TRUE(false);
    }
  }
}

TEST(GEMV, gemv_v6) {
  const int m = 1024, n = 1024;
  std::vector<float> mat_a(m * n, 0.0f);
  std::vector<float> vec_x(n, 0.0f);
  std::vector<float> vec_y(m, 0.0f);
  std::vector<float> cutlass_vec_y(m, 0.0f);
  const float alpha = 1.0f, beta = 0.00f;

  init_random(mat_a, 1.0f, 1.0f);
  init_random(vec_x,1.0f, 1.0f);
  init_random(vec_y,1.0f, 1.0f);

  std::copy(vec_y.begin(), vec_y.end(), cutlass_vec_y.begin());

  gemv_v6(mat_a.data(), vec_x.data(), vec_y.data(), m, n, alpha, beta);
  cutlass_gemv_fp32(mat_a.data(), vec_x.data(), cutlass_vec_y.data(), m, n,
                    alpha, beta);

  const float tolerance = 0.01f;
  for (int i = 0; i < m; ++i) {
    // std::cout << cutlass_vec_y[i] << " " << vec_y[i] << "\n";
    if (std::abs(cutlass_vec_y[i] - vec_y[i]) >= tolerance) {
      std::cout << std::format("cutlass {}, gemv_v6 {}\n", cutlass_vec_y[i],
                               vec_y[i]);
      ASSERT_TRUE(false);
    }
  }
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
  return 0;
}
