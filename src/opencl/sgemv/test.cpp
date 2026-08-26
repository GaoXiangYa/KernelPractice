#include <vector>
#include "sgemv.h"
#include "test_utils.h"

// CPU reference: y[M] = alpha * A[M,N] * x[N] + beta * y[M]
static void sgemv_ref(const float* A, const float* x, float* y, int M, int N,
                      float alpha = 1.0f, float beta = 0.0f) {
  for (int m = 0; m < M; ++m) {
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
      sum += A[m * N + n] * x[n];
    }
    y[m] = alpha * sum + beta * y[m];
  }
}

using SgemvFunc = void (*)(const float*, const float*, float*, int, int, float,
                           float);

struct SgemvVariant {
  const char* name;
  SgemvFunc func;
  float alpha = 1.0f;
  float beta = 0.0f;
};

class SgemvTest : public ::testing::TestWithParam<SgemvVariant> {
protected:
  static constexpr int M = 4096, N = 8192;
  std::vector<float> A, x, y_cpu, y_ocl;

  void SetUp() override {
    A = random_vec(M * N);
    x = random_vec(N);
    y_cpu = random_vec(M);
    y_ocl = y_cpu;
  }
};

TEST_P(SgemvTest, Correctness) {
  auto [name, func, alpha, beta] = GetParam();
  sgemv_ref(A.data(), x.data(), y_cpu.data(), M, N, alpha, beta);
  func(A.data(), x.data(), y_ocl.data(), M, N, alpha, beta);
  expect_near(y_ocl, y_cpu, 1e-3f);
}

INSTANTIATE_TEST_SUITE_P(
    Variants, SgemvTest,
    ::testing::Values(SgemvVariant{"v0", sgemv_v0, 1.0f, 0.0f},
                      SgemvVariant{"v0_beta", sgemv_v0, 0.5f, 0.5f},
                      SgemvVariant{"v1", sgemv_v1, 1.0f, 0.0f},
                      SgemvVariant{"v1_beta", sgemv_v1, 0.5f, 0.5f},
                      SgemvVariant{"v2", sgemv_v1, 1.0f, 0.0f},
                      SgemvVariant{"v2_beta", sgemv_v1, 0.5f, 0.5f}
                      ),
    [](const auto& info) { return info.param.name; });
