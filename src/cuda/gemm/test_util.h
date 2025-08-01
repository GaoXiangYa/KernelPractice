#include <format>
#include <iostream>
#include <random>
#include <vector>

template <typename T>
void init_random_matrix(std::vector<T> &matrix,
                        T min = std::is_integral<T>::value ? 0 : T(0.0),
                        T max = std::is_integral<T>::value ? 100 : T(1.0)) {
  static_assert(std::is_arithmetic<T>::value,
                "Matrix elements must be numeric types");

  std::random_device rd;
  std::mt19937 rng(rd());

  // 根据类型选择分布
  if constexpr (std::is_integral<T>::value) {
    std::uniform_int_distribution<T> dist(min, max);
    for (auto &element : matrix) {
      element = dist(rng);
    }
  } else {
    std::uniform_real_distribution<T> dist(min, max);
    for (auto &element : matrix) {
      element = dist(rng);
    }
  }
}
// a[m, n] * b[n, k] = c[m, k]
inline void ref_matmul(const std::vector<float> &a,
                        const std::vector<float> &b, std::vector<float> &c,
                        int m, int n, int k) {

  for (int i = 0; i < m; ++i) {
    for (int p = 0; p < k; ++p) {
      for (int j = 0; j < n; ++j) {
        c[i * k + p] += a[i * n + j] * b[j * k + p];
      }
    }
  }
}

inline void print_matmul(const float *matrix, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (j == 0)
        std::cout << std::format("[ {},", matrix[i * n + j]);
      else if (j == n - 1)
        std::cout << std::format("{} ]\n", matrix[i * n + j]);
      else
        std::cout << std::format("{}, ", matrix[i * n + j]);
    }
  }
}
