#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>
#include "gemm.cuh"

void ref_gemm_torch(const std::vector<float>& a, const std::vector<float>& b,
                    std::vector<float>& c, int M, int N, int K) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  auto ta = torch::from_blob((void*) a.data(), {M, K}, opts);
  auto tb = torch::from_blob((void*) b.data(), {K, N}, opts);

  auto tc = torch::matmul(ta, tb);

  std::memcpy(c.data(), tc.data_ptr<float>(), M * N * sizeof(float));
}

void test_gemm_v0(int M, int N, int K) {
  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c_ref(M * N, 0.0f);
  std::vector<float> c_cuda(M * N, 0.0f);

  for (auto& v : a)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;
  for (auto& v : b)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;

  ref_gemm_torch(a, b, c_ref, M, N, K);
  gemm_v0(a.data(), b.data(), c_cuda.data(), M, N, K);

  constexpr float kEpsilon = 1e-3f;
  for (int i = 0; i < M * N; ++i) {
    ASSERT_NEAR(c_ref[i], c_cuda[i], kEpsilon) << "Mismatch at index " << i;
  }
}

TEST(GemmTest, v0_small) {
  test_gemm_v0(64, 64, 64);
}

TEST(GemmTest, v0_medium) {
  test_gemm_v0(256, 256, 256);
}

TEST(GemmTest, v0_large) {
  test_gemm_v0(1024, 256, 128);
}

TEST(GemmTest, v0_skinny) {
  test_gemm_v0(1024, 16, 256);
}

void test_gemm_v1(int M, int N, int K) {
  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c_ref(M * N, 0.0f);
  std::vector<float> c_cuda(M * N, 0.0f);

  for (auto& v : a)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;
  for (auto& v : b)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;

  ref_gemm_torch(a, b, c_ref, M, N, K);
  gemm_v1(a.data(), b.data(), c_cuda.data(), M, N, K);

  constexpr float kEpsilon = 1e-3f;
  for (int i = 0; i < M * N; ++i) {
    ASSERT_NEAR(c_ref[i], c_cuda[i], kEpsilon) << "Mismatch at index " << i;
  }
}

TEST(GemmTest, v1_small) {
  test_gemm_v1(64, 64, 64);
}

TEST(GemmTest, v1_medium) {
  test_gemm_v1(256, 256, 256);
}

TEST(GemmTest, v1_large) {
  test_gemm_v1(1024, 256, 128);
}

TEST(GemmTest, v1_skinny) {
  test_gemm_v1(1024, 16, 256);
}

void test_gemm_v2(int M, int N, int K) {
  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c_ref(M * N, 0.0f);
  std::vector<float> c_cuda(M * N, 0.0f);

  for (auto& v : a)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;
  for (auto& v : b)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;

  ref_gemm_torch(a, b, c_ref, M, N, K);
  gemm_v1(a.data(), b.data(), c_cuda.data(), M, N, K);

  constexpr float kEpsilon = 1e-3f;
  for (int i = 0; i < M * N; ++i) {
    ASSERT_NEAR(c_ref[i], c_cuda[i], kEpsilon) << "Mismatch at index " << i;
  }
}

TEST(GemmTest, v2_small) {
  test_gemm_v2(64, 64, 64);
}

TEST(GemmTest, v2_medium) {
  test_gemm_v2(256, 256, 256);
}

TEST(GemmTest, v2_large) {
  test_gemm_v2(1024, 256, 128);
}

TEST(GemmTest, v2_skinny) {
  test_gemm_v2(1024, 16, 256);
}

void test_gemm_v3(int M, int N, int K) {
  std::vector<float> a(M * K);
  std::vector<float> b(K * N);
  std::vector<float> c_ref(M * N, 0.0f);
  std::vector<float> c_cuda(M * N, 0.0f);

  for (auto& v : a)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;
  for (auto& v : b)
    v = float(rand()) / RAND_MAX * 2.0f - 1.0f;

  ref_gemm_torch(a, b, c_ref, M, N, K);
  gemm_v1(a.data(), b.data(), c_cuda.data(), M, N, K);

  constexpr float kEpsilon = 1e-3f;
  for (int i = 0; i < M * N; ++i) {
    ASSERT_NEAR(c_ref[i], c_cuda[i], kEpsilon) << "Mismatch at index " << i;
  }
}

TEST(GemmTest, v3_small) {
  test_gemm_v3(64, 64, 64);
}

TEST(GemmTest, v3_medium) {
  test_gemm_v3(256, 256, 256);
}

TEST(GemmTest, v3_large) {
  test_gemm_v3(1024, 256, 128);
}

TEST(GemmTest, v3_skinny) {
  test_gemm_v3(1024, 16, 256);
}
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
