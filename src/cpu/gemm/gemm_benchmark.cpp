#include "gemm.h"
#include "util.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <span>
#include <unordered_map>

using gemm_func = std::function<void(float *, float *, float *, int, int, int)>;

std::unordered_map<std::string, gemm_func> gemm_map = {
    {"gemm_v0", gemm_v0},
    {"gemm_blas", gemm_blas},
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
    {"gemm_4x4block_v11", gemm_4x4block_v11},
    {"gemm_4x8block_v12", gemm_4x8block_v12},
    {"gemm_4x8block_v13", gemm_4x8block_v13},
    {"gemm_4x8block_v14", gemm_4x8block_v14},
    {"gemm_4x8block_v15", gemm_4x8block_v15},
    {"gemm_4x8block_v16", gemm_4x8block_v16},
    {"gemm_4x8block_v17", gemm_4x8block_v17},
    {"gemm_4x8block_v18", gemm_4x8block_v18},
    {"gemm_4x8block_v19", gemm_4x8block_v19},
    {"gemm_4x8block_v20", gemm_4x8block_v20},
    {"gemm_omp_v18", gemm_omp_v18},
    {"gemm_omp_v19", gemm_omp_v19},
};

double gflops(int M, int N, int K, double seconds) {
  double flops = 2.0 * M * N * K;
  return flops / (seconds * 1e9);
}

void warmUp() {
  const int M = 128, N = 128, K = 128;
  const int ALIGNMENT = 32;
  float *A = reinterpret_cast<float *>(
      std::aligned_alloc(ALIGNMENT, M * K * sizeof(float)));
  float *B = reinterpret_cast<float *>(
      std::aligned_alloc(ALIGNMENT, K * N * sizeof(float)));
  float *C = reinterpret_cast<float *>(
      std::aligned_alloc(ALIGNMENT, M * N * sizeof(float)));
  const int repeat = 3;

  for (int i = 0; i < repeat; ++i) {
    gemm_blas(A, B, C, M, N, K);
  }

  std::free(A);
  std::free(B);
  std::free(C);
}

template <typename Func>
void benchmark(std::ofstream &file, const std::string &name, Func f, float *A,
               float *B, float *C, int m, int n, int k) {
  const int ALIGNMENT = 32;
  const int REPEAT = 100;

  A = reinterpret_cast<float *>(
      std::aligned_alloc(ALIGNMENT, m * k * sizeof(float)));
  initMatrix(A, m, k, -1.0f, 1.0f);
  B = reinterpret_cast<float *>(
      std::aligned_alloc(ALIGNMENT, k * n * sizeof(float)));
  initMatrix(B, k, n, -1.0f, 1.0f);
  C = reinterpret_cast<float *>(
      std::aligned_alloc(ALIGNMENT, m * n * sizeof(float)));
  initMatrix(C, m, n, -1.0f, 1.0f);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < REPEAT; ++i) {
    f(A, B, C, m, n, k);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  file << "[" << name << "] [m, n, k]: " << m
       << " Time: " << elapsed.count() * 1000 / REPEAT
       << " ms, GFLOPS: " << gflops(m, n, k, elapsed.count() / REPEAT) << "\n";

  std::free(A);
  std::free(B);
  std::free(C);
  A = nullptr;
  B = nullptr;
  C = nullptr;
}

void launchBenchmark(const std::string &name) {
  gemm_func func = gemm_map[name];
  float *A = nullptr;
  float *B = nullptr;
  float *C = nullptr;
  std::ofstream gemm_file(name + ".csv", std::ios::out);

  warmUp();

  for (int stride = 16; stride <= 1024; stride += 16) {
    benchmark<gemm_func>(gemm_file, name, func, A, B, C, stride, stride,
                         stride);
  }
}

int main(int argc, const char *argv[]) {
  std::span<const char *> arg_span(argv, argc);
  for (int i = 1; i < arg_span.size(); ++i) {
    launchBenchmark(arg_span[i]);
  }
  return 0;
}
