#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>
#include "gemm.cuh"

using GemmKernel = void (*)(const float*, const float*, float*, int, int, int);

// ---------------------------------------------------------------------------
// reference (libtorch)
// ---------------------------------------------------------------------------
static void ref_gemm(const std::vector<float>& a, const std::vector<float>& b,
                     std::vector<float>& c, int M, int N, int K) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  auto ta = torch::from_blob((void*) a.data(), {M, K}, opts);
  auto tb = torch::from_blob((void*) b.data(), {K, N}, opts);
  auto tc = torch::matmul(ta, tb);
  std::memcpy(c.data(), tc.data_ptr<float>(), M * N * sizeof(float));
}

// ---------------------------------------------------------------------------
// single test helper — all kernels share this
// ---------------------------------------------------------------------------
static void test_gemm(GemmKernel kernel, int M, int N, int K) {
  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c_ref(M * N, 0.0f);
  std::vector<float> c_cuda(M * N, 0.0f);

  for (auto& v : a)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;
  for (auto& v : b)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;

  ref_gemm(a, b, c_ref, M, N, K);
  kernel(a.data(), b.data(), c_cuda.data(), M, N, K);

  constexpr float kEpsilon = 1e-3f;
  for (int i = 0; i < M * N; ++i) {
    ASSERT_NEAR(c_ref[i], c_cuda[i], kEpsilon) << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// macro to stamp out TEST cases
// ---------------------------------------------------------------------------
#define GEMM_TEST(kernel, name, M, N, K) \
  TEST(GemmTest, name) {                 \
    test_gemm(kernel, M, N, K);          \
  }

// v0
GEMM_TEST(gemm_v0, v0_square_small, 64, 64, 64)
GEMM_TEST(gemm_v0, v0_square_medium, 256, 256, 256)
GEMM_TEST(gemm_v0, v0_rect, 1024, 256, 128)
GEMM_TEST(gemm_v0, v0_skinny, 1024, 16, 256)

// v1
GEMM_TEST(gemm_v1, v1_square_small, 64, 64, 64)
GEMM_TEST(gemm_v1, v1_square_medium, 256, 256, 256)
GEMM_TEST(gemm_v1, v1_rect, 1024, 256, 128)
GEMM_TEST(gemm_v1, v1_skinny, 1024, 16, 256)

// v2
GEMM_TEST(gemm_v2, v2_square_small, 64, 64, 64)
GEMM_TEST(gemm_v2, v2_square_medium, 256, 256, 256)
GEMM_TEST(gemm_v2, v2_rect, 1024, 256, 128)
GEMM_TEST(gemm_v2, v2_skinny, 1024, 16, 256)

// v3
GEMM_TEST(gemm_v3, v3_square_small, 64, 64, 64)
GEMM_TEST(gemm_v3, v3_square_medium, 256, 256, 256)
GEMM_TEST(gemm_v3, v3_rect, 1024, 256, 128)
GEMM_TEST(gemm_v3, v3_skinny, 1024, 16, 256)

// v4
GEMM_TEST(gemm_v4, v4_square_small, 64, 64, 64)
GEMM_TEST(gemm_v4, v4_square_medium, 256, 256, 256)
GEMM_TEST(gemm_v4, v4_rect, 1024, 256, 128)
GEMM_TEST(gemm_v4, v4_skinny, 1024, 16, 256)

// v5
GEMM_TEST(gemm_v5, v5_square_small, 64, 64, 64)
GEMM_TEST(gemm_v5, v5_square_medium, 256, 256, 256)
GEMM_TEST(gemm_v5, v5_rect, 1024, 256, 128)
GEMM_TEST(gemm_v5, v5_skinny, 1024, 16, 256)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
