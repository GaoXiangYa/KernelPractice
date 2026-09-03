#include "sgemv_q4_1.h"
#include "test_utils.h"
#include <cmath>
#include <cstring>
#include <vector>

// ===========================================================================
// Q4_1 packing (host) — same convention as gemm_q4_1
// ===========================================================================
static void pack_q4_1(const float* src, int k, int block_k,
                      unsigned char* dst) {
  float d = 0.0f, m = 0.0f;
  for (int i = 0; i < k; ++i) {
    float v = src[i];
    m = fminf(m, v);
    d = fmaxf(d, v - m);
  }
  if (d == 0.0f) d = 1.0f;
  memcpy(dst, &d, sizeof(float));
  dst += sizeof(float);
  memcpy(dst, &m, sizeof(float));
  dst += sizeof(float);
  for (int i = 0; i < k; i += 2) {
    int q0 = (int) roundf((src[i] - m) / d * 15.0f);
    int q1 = (i + 1 < k) ? (int) roundf((src[i + 1] - m) / d * 15.0f) : 0;
    q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
    q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
    *dst++ = (unsigned char) (q0 | (q1 << 4));
  }
}

// ===========================================================================
// CPU reference: y = dequant(A_q4) * x  (float accumulate, random data in
// [-1, 1]; Q4_1 rounding keeps results within ~5.0 of exact at these sizes)
// ===========================================================================
static void sgemv_q4_1_ref(const unsigned char* A_q4, const float* x,
                           float* y, int M, int N, int block_k, float alpha,
                           float beta) {
  int blocks_per_row = N / block_k;
  int block_bytes = (int) (sizeof(float) * 2 + block_k / 2);

  for (int r = 0; r < M; ++r) {
    const unsigned char* row = A_q4 + r * blocks_per_row * block_bytes;
    float sum = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
      const unsigned char* blk = row + b * block_bytes;
      float d, m;
      memcpy(&d, blk, sizeof(float));
      memcpy(&m, blk + sizeof(float), sizeof(float));
      const unsigned char* qs = blk + sizeof(float) * 2;
      for (int i = 0; i < block_k; ++i) {
        unsigned char packed = qs[i / 2];
        int nib = (i & 1) ? (packed >> 4) : (packed & 0x0F);
        sum += ((float) nib * d + m) * x[b * block_k + i];
      }
    }
    y[r] = alpha * sum + beta * y[r];
  }
}

// ===========================================================================
// Variant registry — add new versions here
// ===========================================================================
using SgemvQ4Func = void (*)(const unsigned char*, const float*, float*, int,
                             int, int, float, float);

struct Variant {
  const char* name;
  SgemvQ4Func func;
};

static const Variant kVariants[] = {
    {"v0", sgemv_q4_1_v0},
};

// ===========================================================================
// Test fixture
// ===========================================================================
class SgemvQ4Test : public ::testing::Test {
protected:
  void run_case(int M, int N, int block_k = 32, float alpha = 1.0f,
                float beta = 0.0f, float eps = 5.0f) {
    std::vector<float> A_f32(M * N);
    for (size_t i = 0; i < A_f32.size(); ++i)
      A_f32[i] = (float) (rand() % 100) / 50.0f - 1.0f;

    std::vector<float> x_f32(N);
    for (size_t i = 0; i < x_f32.size(); ++i)
      x_f32[i] = (float) (rand() % 100) / 50.0f - 1.0f;

    int blocks_per_row = N / block_k;
    int block_bytes = (int) (sizeof(float) * 2 + block_k / 2);
    std::vector<unsigned char> A_q4(M * blocks_per_row * block_bytes);
    for (int r = 0; r < M; ++r)
      for (int b = 0; b < blocks_per_row; ++b)
        pack_q4_1(&A_f32[r * N + b * block_k], block_k, block_k,
                  &A_q4[r * blocks_per_row * block_bytes + b * block_bytes]);

    std::vector<float> y_cpu(M), y_ocl(M);
    for (int r = 0; r < M; ++r) {
      y_cpu[r] = (float) (rand() % 100) / 50.0f - 1.0f;
      y_ocl[r] = y_cpu[r];
    }

    sgemv_q4_1_ref(A_q4.data(), x_f32.data(), y_cpu.data(), M, N, block_k,
                   alpha, beta);

    for (auto& [name, func] : kVariants) {
      SCOPED_TRACE(name);
      std::vector<float> y_out = y_ocl;
      func(A_q4.data(), x_f32.data(), y_out.data(), M, N, block_k, alpha,
           beta);
      expect_near(y_out, y_cpu, eps);
    }
  }
};

// ===========================================================================
// Test cases — each runs against ALL registered variant kernels
// ===========================================================================
TEST_F(SgemvQ4Test, Tiny)   { run_case(8, 64); }
TEST_F(SgemvQ4Test, Small)  { run_case(64, 256); }
TEST_F(SgemvQ4Test, Medium) { run_case(256, 512); }
TEST_F(SgemvQ4Test, Wide)   { run_case(64, 2048); }
TEST_F(SgemvQ4Test, AlphaBeta) { run_case(64, 256, 32, 2.0f, 0.5f); }
TEST_F(SgemvQ4Test, NonDefaultBlockK) { run_case(64, 256, 64); }
