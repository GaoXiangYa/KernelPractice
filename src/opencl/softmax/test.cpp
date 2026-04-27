#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include "softmax.h"

// Tensor shape struct
struct SoftmaxShape {
  int B, H, S_q, S_k;
};

// CPU reference softmax: normalize along the last dimension (seq_k)
std::vector<float> cpuSoftmax(const float* input, const float* mask,
                              float scale, int is_causal, int B, int H, int S_q,
                              int S_k) {
  std::vector<float> output(B * H * S_q * S_k);
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int q = 0; q < S_q; ++q) {
        // base offset
        int base = ((b * H + h) * S_q + q) * S_k;
        float max_val = -INFINITY;
        // find max value
        for (int k = 0; k < S_k; ++k) {
          float val = input[base + k] * scale;
          if (mask != nullptr) {
            val += mask[q * S_k + k];  // mask shape [S_q, S_k]
          }
          if (is_causal && k > q) {
            val = -INFINITY;
          }
          max_val = fmaxf(max_val, val);
        }
        // exp and sum
        float sum = 0.0f;
        for (int k = 0; k < S_k; ++k) {
          float val = input[base + k] * scale;
          if (mask != nullptr) {
            val += mask[q * S_k + k];
          }
          if (is_causal && k > q) {
            val = -INFINITY;
          }
          output[base + k] = expf(val - max_val);
          sum += output[base + k];
        }
        // normalize
        for (int k = 0; k < S_k; ++k) {
          output[base + k] /= sum;
        }
      }
    }
  }
  return output;
}

// Test fixture: manage device memory and default params
class SoftmaxTest : public ::testing::Test {
protected:
  void SetUp() override {
    // allocate large memory if needed; tests allocate dynamically
  }
  void TearDown() override {}

  // Helper: run GPU version and compare with CPU
  void runTest(const std::vector<float>& h_input,
               const std::vector<float>& h_mask, bool use_mask, float scale,
               int is_causal, int B, int H, int S_q, int S_k) {
    // compute total element count
    size_t total = B * H * S_q * S_k;
    ASSERT_EQ(h_input.size(), total);
    std::vector<float> h_output(total);
    // launch kernel
    launchSoftmaxV0(h_output.data(), h_input.data(),
                    use_mask ? h_mask.data() : nullptr, B, H, S_q, S_k, scale,
                    is_causal);

    // CPU reference result
    auto ref = cpuSoftmax(h_input.data(), use_mask ? h_mask.data() : nullptr,
                          scale, is_causal, B, H, S_q, S_k);

    // compare error
    for (size_t i = 0; i < total; ++i) {
      ASSERT_NEAR(h_output[i], ref[i], 1e-2f) << "Mismatch at index " << i;
    }
  }
};

// ---- test cases ----

// 1. simplest case: single batch, single head, small sequence, no mask, no
// causal
TEST_F(SoftmaxTest, BasicSmallNoMask) {
  int B = 1, H = 1, S_q = 4, S_k = 4;
  float scale = 1.0f;  // no scaling
  std::vector<float> input = {0.1f,  0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f,
                              -1.0f, 0.0f, 1.0f, 2.0f, 5.0f, 1.0f, 2.0f, 0.5f};
  runTest(input, {}, false, scale, 0, B, H, S_q, S_k);
}

// 2. with scaling factor
TEST_F(SoftmaxTest, WithScale) {
  int B = 1, H = 2, S_q = 3, S_k = 4;
  float scale = 1.0f / sqrtf(8.0f);  // simulate d_k=8
  size_t total = B * H * S_q * S_k;
  std::vector<float> input(total);
  for (size_t i = 0; i < total; ++i)
    input[i] = (rand() % 100) / 100.0f;
  runTest(input, {}, false, scale, false, B, H, S_q, S_k);
}

// 3. with mask, hide certain positions
TEST_F(SoftmaxTest, WithMask) {
  int B = 1, H = 1, S_q = 3, S_k = 4;
  float scale = 1.0f;
  std::vector<float> input = {
      1.0f, 2.0f,  3.0f,  4.0f,  // query 0
      5.0f, 6.0f,  7.0f,  8.0f,  // query 1
      9.0f, 10.0f, 11.0f, 12.0f  // query 2
  };
  // mask: [S_q, S_k], -inf used for masking
  std::vector<float> mask = {
      0.0f,      0.0f,
      -INFINITY, 0.0f,  // mask key 2 for query 0
      0.0f,      -INFINITY,
      0.0f,      -INFINITY,  // mask key 1 and key 3 for query 1
      -INFINITY, 0.0f,
      0.0f,      0.0f  // mask key 0 for query 2
  };
  // note: -INFINITY is replaced with a large negative number, e.g. -1e10f;
  // production code may use -1e10f to represent -inf
  // here replaced with -1e10f for testing, allow numerical differences
  float neg_inf = -1e10f;
  for (auto& v : mask) {
    if (std::isinf(v) && v < 0) {
      v = neg_inf;
    }
  }

  runTest(input, mask, true, scale, false, B, H, S_q, S_k);
}

// 4. causal mask: upper triangle zero (probability=0), no external mask needed 
TEST_F(SoftmaxTest, CausalMask) {
  int B = 1, H = 1, S_q = 4, S_k = 4;
  float scale = 1.0f;
  std::vector<float> input = {0.1f,  0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f,
                              -1.0f, 0.0f, 1.0f, 2.0f, 5.0f, 1.0f, 2.0f, 0.5f};
  runTest(input, {}, false, scale, true, B, H, S_q, S_k);
}

// 5. multi-batch multi-head test
TEST_F(SoftmaxTest, MultiBatchHead) {
  int B = 2, H = 3, S_q = 5, S_k = 7;
  float scale = 1.0f / sqrtf(16.0f);
  size_t total = B * H * S_q * S_k;
  std::vector<float> input(total);
  for (size_t i = 0; i < total; ++i)
    input[i] = (rand() % 100) / 50.0f - 1.0f;
  runTest(input, {}, false, scale, false, B, H, S_q, S_k);
}

// 6. different query and key lengths (no causal)
TEST_F(SoftmaxTest, DifferentSeqLen) {
  int B = 1, H = 2, S_q = 3, S_k = 5;
  float scale = 1.0f / sqrtf(10.0f);
  size_t total = B * H * S_q * S_k;
  std::vector<float> input(total);
  for (size_t i = 0; i < total; ++i)
    input[i] = (rand() % 100) / 80.0f;
  runTest(input, {}, false, scale, false, B, H, S_q, S_k);
}

// 7. large dimension stress test (but quick)
TEST_F(SoftmaxTest, LargeDimensions) {
  int B = 2, H = 4, S_q = 32, S_k = 64;
  float scale = 1.0f / sqrtf(32.0f);
  size_t total = B * H * S_q * S_k;
  std::vector<float> input(total);
  for (size_t i = 0; i < total; ++i)
    input[i] = (rand() % 100) / 25.0f;
  runTest(input, {}, false, scale, false, B, H, S_q, S_k);
}

// 8. nullptr mask pointer, ensure no crash
TEST_F(SoftmaxTest, NullMaskPtr) {
  int B = 1, H = 1, S_q = 2, S_k = 3;
  float scale = 1.0f;
  std::vector<float> input = {1, 2, 3, 4, 5, 6};
  runTest(input, {}, false, scale, false, B, H, S_q, S_k);
}

// 9. kernel numerical stability: large input values
TEST_F(SoftmaxTest, HandlingLargeValues) {
  int B = 1, H = 1, S_q = 2, S_k = 4;
  float scale = 1.0f;
  std::vector<float> input = {1000.0f, 1000.0f, 1000.0f, 1000.0f,  // uniform
                              1e5f,    -1e5f,   0.0f,    0.0f};
  runTest(input, {}, false, scale, false, B, H, S_q, S_k);
}
