#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "rmsnorm.h"
#include "test_utils.h"

// ===========================================================================
// LibTorch reference
// ===========================================================================
static std::vector<float> torch_rmsnorm(const float* input, const float* weight,
                                        int N, int D, float epsilon) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto x = torch::from_blob(const_cast<float*>(input), {N, D}, opts).clone();
    auto w = torch::from_blob(const_cast<float*>(weight), {D}, opts);

    auto sq  = x * x;
    auto rms = torch::sqrt(sq.mean(-1, true) + epsilon);
    auto out = (x / rms) * w;

    auto result = out.contiguous();
    auto* ptr   = result.data_ptr<float>();
    return std::vector<float>(ptr, ptr + result.numel());
}

// ===========================================================================
// Variant registry — add new versions here
// ===========================================================================
using RmsnormFunc = void (*)(const std::vector<float>&, const std::vector<float>&,
                             std::vector<float>&, int, int, float);

struct Variant {
    const char*  name;
    RmsnormFunc  func;
};

static const Variant kVariants[] = {
    // {"v0", launch_rmsnorm_v0},
    // {"v1", launch_rmsnorm_v1},
    {"v2", launch_rmsnorm_v2},   // future
};

// ===========================================================================
// Test fixture
// ===========================================================================
class RMSNormTest : public ::testing::Test {
protected:
    void run_case(int N, int D, float epsilon = 1e-5f,
                  float eps_cmp = 1e-3f) {
        auto input  = random_vec(N * D, -2.0f, 2.0f);
        auto weight = random_vec(D,      0.5f, 1.5f);

        auto ref = torch_rmsnorm(input.data(), weight.data(), N, D, epsilon);

        for (auto& [name, func] : kVariants) {
            SCOPED_TRACE(name);
            std::vector<float> output(N * D, -99.0f);
            func(input, weight, output, N, D, epsilon);
            expect_near(output, ref, eps_cmp);
        }
    }
};

// ===========================================================================
// Test cases
// ===========================================================================

TEST_F(RMSNormTest, SingleRowSmall)     { run_case(1, 16); }
TEST_F(RMSNormTest, Square)             { run_case(4, 64); }
TEST_F(RMSNormTest, ManyRows)           { run_case(32, 128); }
TEST_F(RMSNormTest, LargeD)             { run_case(8, 1024); }
TEST_F(RMSNormTest, LargeNandD)         { run_case(64, 512); }

TEST_F(RMSNormTest, AllZeroInput) {
    int N = 2, D = 32;
    std::vector<float> input(N * D, 0.0f);
    std::vector<float> weight(D, 1.0f);
    auto ref = torch_rmsnorm(input.data(), weight.data(), N, D, 1e-5f);

    for (auto& [name, func] : kVariants) {
        SCOPED_TRACE(name);
        std::vector<float> output(N * D, -1.0f);
        func(input, weight, output, N, D, 1e-5f);
        expect_near(output, ref, 1e-4f);
    }
}

TEST_F(RMSNormTest, NumericalStability) {
    std::vector<float> input = {1e5f, -1e5f, 0.0f, 1.0f, -1.0f, 2.0f,
                                0.5f, -0.5f,  3.0f, -3.0f, 4.0f, -4.0f};
    std::vector<float> weight(12, 1.0f);
    auto ref = torch_rmsnorm(input.data(), weight.data(), /*N=*/1, /*D=*/12, 1e-5f);

    for (auto& [name, func] : kVariants) {
        SCOPED_TRACE(name);
        std::vector<float> output(12, -1.0f);
        func(input, weight, output, 1, 12, 1e-5f);
        expect_near(output, ref, 1e-2f);
    }
}
