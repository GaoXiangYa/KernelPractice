#pragma once

#include <cstdint>

struct FlashAttentionParams {
  // Q, K, V
  const void* q;
  const void* k;
  const void* v;

  // Output
  void* o;

  // Tensor shape
  int batch;
  int num_heads;
  int num_kv_heads;

  int seq_len_q;
  int seq_len_k;

  int head_dim;

  // Stride -- units are ELEMENTS, not bytes (make_flash_attn_params sets them
  // for contiguous row-major tensors; the kernel loaders index T* by these).
  int64_t q_stride_b;
  int64_t q_stride_h;
  int64_t q_stride_s;

  int64_t k_stride_b;
  int64_t k_stride_h;
  int64_t k_stride_s;

  int64_t v_stride_b;
  int64_t v_stride_h;
  int64_t v_stride_s;

  int64_t o_stride_b;
  int64_t o_stride_h;
  int64_t o_stride_s;

  // Attention
  float softmax_scale;

  bool causal;

  // Optional
  bool return_softmax;
};

// Host helper: build a params bundle describing contiguous row-major tensors.
inline FlashAttentionParams make_flash_attn_params(
    const void* q, const void* k, const void* v, void* o,
    int batch, int num_heads, int seq_len_q, int seq_len_k, int head_dim,
    float softmax_scale, bool causal) {
  FlashAttentionParams p{};
  p.q = q;
  p.k = k;
  p.v = v;
  p.o = o;
  p.batch = batch;
  p.num_heads = num_heads;
  p.num_kv_heads = num_heads;
  p.seq_len_q = seq_len_q;
  p.seq_len_k = seq_len_k;
  p.head_dim = head_dim;
  const int64_t sq = seq_len_q, sk = seq_len_k, hd = head_dim;
  p.q_stride_b = num_heads * sq * hd;  p.q_stride_h = sq * hd;  p.q_stride_s = hd;
  p.k_stride_b = num_heads * sk * hd;  p.k_stride_h = sk * hd;  p.k_stride_s = hd;
  p.v_stride_b = num_heads * sk * hd;  p.v_stride_h = sk * hd;  p.v_stride_s = hd;
  p.o_stride_b = num_heads * sq * hd;  p.o_stride_h = sq * hd;  p.o_stride_s = hd;
  p.softmax_scale = softmax_scale;
  p.causal = causal;
  p.return_softmax = false;
  return p;
}