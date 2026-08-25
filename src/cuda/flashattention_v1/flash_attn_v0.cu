#include "flashattention_v1.h"
#include "util.h"

static __global__ void flash_attn_v0_kernel(float* O, int total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total)
    O[i] = 0.0f;
}

void flash_attn_v0(const float* Q, const float* K, const float* V, float* O,
                   int B, int H, int N, int d, bool causal) {
  (void) causal;  // TODO(M1): causal mask
  const int total = B * H * N * d;

  float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
  CHECK_CUDA(cudaMalloc(&dQ, total * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dK, total * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dV, total * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dO, total * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dQ, Q, total * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK, K, total * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV, V, total * sizeof(float), cudaMemcpyHostToDevice));

  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  flash_attn_v0_kernel<<<blocks, kThreads>>>(dO, total);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O, dO, total * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dQ));
  CHECK_CUDA(cudaFree(dK));
  CHECK_CUDA(cudaFree(dV));
  CHECK_CUDA(cudaFree(dO));
}
