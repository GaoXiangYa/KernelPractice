#include "flash_attn.cuh"
#include "flashattention_v1.h"
#include "util.h"

// TODO(M2): online-softmax tiled kernel (IMPLEMENTATION.md M2), stub for now.
static __global__ void flash_attn_v1_kernel(FlashAttentionParams params) {
  const int total =
      params.batch * params.num_heads * params.seq_len_q * params.head_dim;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) reinterpret_cast<float*>(params.o)[i] = 0.0f;
}

void flash_attn_v1(const FlashAttentionParams& params) {
  const int total =
      params.batch * params.num_heads * params.seq_len_q * params.head_dim;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  flash_attn_v1_kernel<<<blocks, kThreads>>>(params);
  CHECK_CUDA(cudaGetLastError());
}

void flash_attn_v1(const float* Q, const float* K, const float* V, float* O,
                   int B, int H, int N, int d, bool causal) {
  // identical host-wrapper shape to flash_attn_v0; builds params and calls
  // flash_attn_v1(p); returns early? No - same full copy path, TODO(M2) kernel.
  const int total = B * H * N * d;
  float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
  CHECK_CUDA(cudaMalloc(&dQ, total * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dK, total * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dV, total * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dO, total * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dQ, Q, total * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK, K, total * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV, V, total * sizeof(float), cudaMemcpyHostToDevice));
  FlashAttentionParams p = make_flash_attn_params(
      dQ, dK, dV, dO, B, H, N, N, d, 1.0f / std::sqrt((float)d), causal);
  flash_attn_v1(p);
  CHECK_CUDA(cudaMemcpy(O, dO, total * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(dQ));
  CHECK_CUDA(cudaFree(dK));
  CHECK_CUDA(cudaFree(dV));
  CHECK_CUDA(cudaFree(dO));
}
