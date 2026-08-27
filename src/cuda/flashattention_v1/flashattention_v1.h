#pragma once
#include "flash_attn.cuh"

// Device-pointer entries: no memory management, launch straight from params.
void flash_attn_v0(const FlashAttentionParams& params);
void flash_attn_v1(const FlashAttentionParams& params);

// Host convenience: contiguous row-major tensors, copies in/out internally.
void flash_attn_v0(const float* Q, const float* K, const float* V, float* O,
                   int B, int H, int N, int d, bool causal);
void flash_attn_v1(const float* Q, const float* K, const float* V, float* O,
                   int B, int H, int N, int d, bool causal);
