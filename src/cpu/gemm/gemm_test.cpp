#include "gemm.h"
#include "util.h"
#include <cstdlib>
#include <cstring>
// #include <format>
#include <functional>
#include <iostream>
#include <map>
#include <span>

using gemm_func = std::function<void(float *, float *, float *, int, int, int)>;

std::map<std::string, gemm_func> gemm_map = {
    {"gemm_v0", gemm_v0},
    {"gemm_v1", gemm_v1},
    {"gemm_v2", gemm_v2},
    {"gemm_v3", gemm_v3},
    {"gemm_v4", gemm_v4},
    {"gemm_v5", gemm_v5},
    {"gemm_v6", gemm_v6},
    {"gemm_v7", gemm_v7},
    {"gemm_v8", gemm_v8},
    {"gemm_4x4block_v3", gemm_4x4block_v3},
    {"gemm_4x4block_v4", gemm_4x4block_v4},
    {"gemm_4x4block_v5", gemm_4x4block_v5},
    {"gemm_4x4block_v6", gemm_4x4block_v6},
    {"gemm_4x4block_v7", gemm_4x4block_v7},
    {"gemm_4x4block_v8", gemm_4x4block_v8},
    {"gemm_4x4block_v9", gemm_4x4block_v9},
    {"gemm_4x4block_v10", gemm_4x4block_v10},
};

template <typename Func> void launchTest(const std::string &gemm_name, Func f) {
  const int m = 256, n = 256, k = 256;
  const float random_min = -1.0f, random_max = 1.0f;

  float *A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));
  float *BLAS_A = reinterpret_cast<float *>(std::malloc(m * k * sizeof(float)));

  initMatrix(A, m, k, random_min, random_max);
  std::memcpy(BLAS_A, A, m * k * sizeof(float));

  float *B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));
  float *BLAS_B = reinterpret_cast<float *>(std::malloc(k * n * sizeof(float)));

  initMatrix(B, k, n, random_min, random_max);
  std::memcpy(BLAS_B, B, k * n * sizeof(float));

  float *C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));
  float *BLAS_C = reinterpret_cast<float *>(std::malloc(m * n * sizeof(float)));

  initMatrix(C, m, n, random_min, random_max);
  std::memcpy(BLAS_C, C, m * n * sizeof(float));

  f(A, B, C, m, n, k);
  gemm_blas(BLAS_A, BLAS_B, BLAS_C, m, n, k);

  const float tolerance = 0.001f;
  for (int i = 0; i < m * n; ++i) {
    if (std::abs(BLAS_C[i] - C[i]) > tolerance) {
      std::cout << gemm_name << " failed!\n";
      std::cout << " openblas is " << BLAS_C[i] << " " << gemm_name << " is "
                << C[i] << "\n";
      // std::cout << std::format("{} failed!\n openblas is {} {} is {}\n",
      //  gemm_name, BLAS_C[i], gemm_name, C[i]);
      return;
    }
  }
  std::cout << gemm_name << " passed!\n";
}

int main(int argc, const char *argv[]) {
  std::span<const char *> arg_span(argv, argc);
  if (arg_span.size() == 1) {
    for (const auto &[gemm_name, gemm_f] : gemm_map) {
      launchTest(gemm_name, gemm_f);
    }
  } else {
    for (int i = 1; i < arg_span.size(); ++i) {
      launchTest(arg_span[i], gemm_map[arg_span[i]]);
    }
  }
  return 0;
}