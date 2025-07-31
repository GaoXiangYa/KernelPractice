#include "test_util.h"
#include <gtest/gtest.h>
#include <vector>

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