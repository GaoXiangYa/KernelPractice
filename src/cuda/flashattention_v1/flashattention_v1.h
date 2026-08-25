#pragma once

void flash_attn_v0(const float* Q, const float* K, const float* V, float* O,
                   int B, int H, int N, int d, bool causal);
void flash_attn_v1(const float* Q, const float* K, const float* V, float* O,
                   int B, int H, int N, int d, bool causal);
