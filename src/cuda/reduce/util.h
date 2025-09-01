#include <vector>
#include <random>

template <typename T>
void init_random(std::vector<T> &matrix,
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

template<typename T = float>
T reduce_ref(const std::vector<T>& input) {
  T init = 0;
  return std::accumulate(input.begin(), input.end(), init);
}