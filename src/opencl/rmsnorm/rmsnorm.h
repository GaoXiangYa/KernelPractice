#pragma once
#include <cstddef>
#include <vector>

// v0 — local memory tree reduce (default local=256, dynamic local mem)
void launch_rmsnorm_v0(const float* input, const float* weight, float* output,
                       int N, int D, float epsilon);
void launch_rmsnorm_v0(const std::vector<float>& input,
                       const std::vector<float>& weight,
                       std::vector<float>& output, int N, int D, float epsilon);

// v1 — float4 + sub_group_reduce  (caller configures global / local)
void launch_rmsnorm_v1(const float* input, const float* weight, float* output,
                       int N, int D, float epsilon);
void launch_rmsnorm_v1(const std::vector<float>& input,
                       const std::vector<float>& weight,
                       std::vector<float>& output, int N, int D, float epsilon);

void launch_rmsnorm_v2(const float* input, const float* weight, float* output,
                       int N, int D, float epsilon);
void launch_rmsnorm_v2(const std::vector<float>& input,
                       const std::vector<float>& weight,
                       std::vector<float>& output, int N, int D, float epsilon);
