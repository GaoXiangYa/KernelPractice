#include "matmul.cuh"
#include "test_util.h"
#include <gtest/gtest.h>
#include <vector>

// TEST(MatMulTest, matmul_native) {
//   const int m = 1024, n = 256, k = 128;
//   std::vector<float> host_a(m * n);
//   std::vector<float> host_b(n * k);
//   std::vector<float> host_c(m * k);

//   init_random_matrix(host_a, -1.0f, 1.0f);
//   init_random_matrix(host_b, -1.0f, 1.0f);
//   init_random_matrix(host_c, 0.0f, 0.0f);

//   float *dev_a = nullptr;
//   float *dev_b = nullptr;
//   float *dev_c = nullptr;

//   auto err = cudaMalloc(dev_a, host_a.size() * sizeof(float));
// }

int main() {
  const int m = 2, n = 2, k = 2;
  std::vector<float> a(m * n);
  std::vector<float> b(m * n);
  std::vector<float> c(m * k);
  init_random_matrix(a, 1.0f, 1.0f);
  init_random_matrix(b, 1.0f, 1.0f);
  ref_matmul(a, b, c, m, n, k);

  print_matmul(c.data(), m, n);
  return 0;
}