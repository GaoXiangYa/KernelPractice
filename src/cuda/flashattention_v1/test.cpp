#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "flashattention_v1.h"

using FlashAttnKernel = void (*)(const float*, const float*, const float*,
                                 float*, int, int, int, int, bool);

// ---------------------------------------------------------------------------
// reference (libtorch eager CPU)
// ---------------------------------------------------------------------------
static void ref_flash_attn(const std::vector<float>& q, const std::vector<float>& k,
                           const std::vector<float>& v, std::vector<float>& o,
                           int B, int H, int N, int d, bool causal) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  auto tq = torch::from_blob((void*)q.data(), {B, H, N, d}, opts);
  auto tk = torch::from_blob((void*)k.data(), {B, H, N, d}, opts);
  auto tv = torch::from_blob((void*)v.data(), {B, H, N, d}, opts);

  torch::Tensor scores = torch::matmul(tq, tk.transpose(-1, -2));
  scores = scores / std::sqrt((double)d);
  if (causal) {
    // strict upper triangle -inf (keep diagonal and below as 0)
    auto mask = torch::triu(torch::full({N, N}, -std::numeric_limits<float>::infinity(), opts), 1);
    scores = scores + mask.unsqueeze(0).unsqueeze(0);
  }
  auto p = torch::softmax(scores, -1);
  auto to = torch::matmul(p, tv);
  std::memcpy(o.data(), to.data_ptr<float>(), B * H * N * d * sizeof(float));
}

// ---------------------------------------------------------------------------
// double-precision naive CPU reference (for M0 self-check, no kernel dependency)
// ---------------------------------------------------------------------------
static void naive_flash_attn_double(const std::vector<float>& q,
                                    const std::vector<float>& k,
                                    const std::vector<float>& v,
                                    std::vector<float>& o, int B, int H, int N,
                                    int d, bool causal) {
  for (int bh = 0; bh < B * H; ++bh) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        // softmax((Q@Kᵀ)/√d) row; fill -inf when causal and j>i
        double m = -std::numeric_limits<double>::infinity();
        std::vector<double> s(N);
        for (int t = 0; t < N; ++t) {
          double acc = 0.0;
          for (int dd = 0; dd < d; ++dd)
            acc += (double)q[(bh * N + i) * d + dd] * (double)k[(bh * N + t) * d + dd];
          s[t] = acc / std::sqrt((double)d);
          if (causal && t > i) s[t] = -std::numeric_limits<double>::infinity();
          if (s[t] > m) m = s[t];
        }
        double sum = 0.0;
        for (int t = 0; t < N; ++t) sum += std::exp(s[t] - m);
        for (int dd = 0; dd < d; ++dd) {
          double acc = 0.0;
          for (int t = 0; t < N; ++t)
            acc += std::exp(s[t] - m) / sum * (double)v[(bh * N + t) * d + dd];
          o[(bh * N + i) * d + dd] = (float)acc;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// single test helper — all kernels share this
// ---------------------------------------------------------------------------
static void test_flash_attn(FlashAttnKernel kernel, int B, int H, int N, int d, bool causal) {
  std::vector<float> q(B * H * N * d), k(B * H * N * d), v(B * H * N * d);
  std::vector<float> o_ref(B * H * N * d), o_cuda(B * H * N * d, 0.0f);
  for (auto& x : q) x = float(rand()) / RAND_MAX * 2.0f - 1.0f;
  for (auto& x : k) x = float(rand()) / RAND_MAX * 2.0f - 1.0f;
  for (auto& x : v) x = float(rand()) / RAND_MAX * 2.0f - 1.0f;

  ref_flash_attn(q, k, v, o_ref, B, H, N, d, causal);
  kernel(q.data(), k.data(), v.data(), o_cuda.data(), B, H, N, d, causal);

  constexpr float kEpsilon = 1e-4f;
  for (int i = 0; i < B * H * N * d; ++i) {
    ASSERT_NEAR(o_ref[i], o_cuda[i], kEpsilon) << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// macro to stamp out TEST cases
// ---------------------------------------------------------------------------
#define FLASH_ATTN_TEST(kernel, name, B, H, N, D, CAUSAL) \
  TEST(FlashAttnTest, name) { test_flash_attn(kernel, B, H, N, D, CAUSAL); }

// v0
FLASH_ATTN_TEST(flash_attn_v0, v0_1x1_64_64_nc, 1, 1, 64, 64, false)
FLASH_ATTN_TEST(flash_attn_v0, v0_1x1_64_64_c, 1, 1, 64, 64, true)
FLASH_ATTN_TEST(flash_attn_v0, v0_1x1_1024_64_nc, 1, 1, 1024, 64, false)
FLASH_ATTN_TEST(flash_attn_v0, v0_1x1_1024_64_c, 1, 1, 1024, 64, true)
FLASH_ATTN_TEST(flash_attn_v0, v0_1x8_512_128_nc, 1, 8, 512, 128, false)
FLASH_ATTN_TEST(flash_attn_v0, v0_1x8_256_128_c, 1, 8, 256, 128, true)
FLASH_ATTN_TEST(flash_attn_v0, v0_1x1_256_32_nc, 1, 1, 256, 32, false)
FLASH_ATTN_TEST(flash_attn_v0, v0_1x1_128_96_c, 1, 1, 128, 96, true)
FLASH_ATTN_TEST(flash_attn_v0, v0_1x1_1_64_nc, 1, 1, 1, 64, false)

// v1
FLASH_ATTN_TEST(flash_attn_v1, v1_1x1_64_64_nc, 1, 1, 64, 64, false)
FLASH_ATTN_TEST(flash_attn_v1, v1_1x1_64_64_c, 1, 1, 64, 64, true)
FLASH_ATTN_TEST(flash_attn_v1, v1_1x1_1024_64_nc, 1, 1, 1024, 64, false)
FLASH_ATTN_TEST(flash_attn_v1, v1_1x1_1024_64_c, 1, 1, 1024, 64, true)
FLASH_ATTN_TEST(flash_attn_v1, v1_1x8_512_128_nc, 1, 8, 512, 128, false)
FLASH_ATTN_TEST(flash_attn_v1, v1_1x8_256_128_c, 1, 8, 256, 128, true)
FLASH_ATTN_TEST(flash_attn_v1, v1_1x1_256_32_nc, 1, 1, 256, 32, false)
FLASH_ATTN_TEST(flash_attn_v1, v1_1x1_128_96_c, 1, 1, 128, 96, true)
FLASH_ATTN_TEST(flash_attn_v1, v1_1x1_1_64_nc, 1, 1, 1, 64, false)

// ---------------------------------------------------------------------------
// M0 self-check: double naive reference vs libtorch reference, proving harness works
// ---------------------------------------------------------------------------
TEST(FlashAttnReference, matches_naive_double) {
  const int B = 1, H = 1, N = 3, d = 4;
  std::vector<float> q(B * H * N * d), k(B * H * N * d), v(B * H * N * d);
  std::vector<float> o_naive(B * H * N * d), o_ref(B * H * N * d);
  for (auto& x : q) x = float(rand()) / RAND_MAX * 2.0f - 1.0f;
  for (auto& x : k) x = float(rand()) / RAND_MAX * 2.0f - 1.0f;
  for (auto& x : v) x = float(rand()) / RAND_MAX * 2.0f - 1.0f;

  naive_flash_attn_double(q, k, v, o_naive, B, H, N, d, false);
  ref_flash_attn(q, k, v, o_ref, B, H, N, d, false);

  constexpr double kEpsilon = 1e-6;
  for (int i = 0; i < B * H * N * d; ++i) {
    ASSERT_NEAR(o_naive[i], o_ref[i], kEpsilon) << "Mismatch at index " << i;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
