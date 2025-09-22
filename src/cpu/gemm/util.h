#include <random>

template <typename T = float>
void initMatrix(float *matrix, int m, int n, T min, T max) {
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<T> dist(min, max);

  for (int i = 0; i < m * n; ++ i) {
    matrix[i] = dist(gen);
  }
}