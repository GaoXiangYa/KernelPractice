#include "matmul.cuh"
#include "test_util.h"
#include <gtest/gtest.h>
#include <vector>

TEST(MatMulTest, matmul_native) {
  const int m = 1024, n = 256, k = 128;
  std::vector<float> host_a(m * n);
  std::vector<float> host_b(n * k);
  std::vector<float> host_cpu_c(m * k, 0.0f);
  std::vector<float> host_gpu_c(m * k, 0.0f);

  init_random_matrix(host_a, -1.0f, 1.0f);
  init_random_matrix(host_b, -1.0f, 1.0f);

  ref_matmul(host_a, host_b, host_cpu_c, m, n, k);
  matmul_native(host_a.data(), host_b.data(), host_gpu_c.data(), m, n, k);

  compare_matrix(host_cpu_c, host_gpu_c);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
  return 0;
}