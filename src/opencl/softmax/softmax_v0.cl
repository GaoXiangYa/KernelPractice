#define MIN(a, b) a < b ? a : b
#define MAX(a, b) a < b ? b : a

__kernel void softmaxKernelV0(
    __global float* restrict output,       // Softmax output (attention weights)
    const __global float* restrict input,  // Attention scores: QK^T, scaled
                                           // or unscaled (determined by
                                           // scale)
    const __global float* restrict mask,   // Mask matrix: can be causal
                                           // mask, padding mask, etc.; can
                                           // be nullptr for no mask
    int batch_size,                        // B: batch size
    int num_heads,                         // H: number of attention heads
    int seq_q,                             // S_q: query sequence length
    int seq_k,                             // S_k: key sequence length
    float scale,   // Scaling factor: 1/sqrt(d_k), pass 1.0f if pre-scaled
    int is_causal  // Whether to automatically generate lower triangular causal
                   // mask (ignore mask input)
) {
  int group0 = get_group_id(0);
  int group1 = get_group_id(1);
  int group2 = get_group_id(2);

  int lid = get_local_id(0);
  int local_size = get_local_size(0);
  int local_size_half = local_size >> 1;

  __local float sdata[1024];

  __global float* pinput = (input + group0 * seq_k + group1 * seq_q * seq_k +
                            group2 * seq_k * seq_q * num_heads);

  __global float* poutput = (output + group0 * seq_k + group1 * seq_q * seq_k +
                             group2 * seq_k * seq_q * num_heads);

  float fmax = -INFINITY;
  for (int i = lid; i < seq_k; i += local_size) {
    if (is_causal && i > group0) {
      fmax = MAX(fmax, pinput[i] * scale);
    } else {
      fmax = MAX(fmax, pinput[i] * scale + (mask == NULL ? 0.0f : mask[i]));
    }
  }
  sdata[lid] = fmax;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int offset = local_size_half; offset >= 1; offset >>= 1) {
    if (lid < offset) {
      sdata[lid] = MAX(sdata[lid], sdata[lid + offset]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  float max_v = sdata[0];

  float sum = 0.0f;
  float exp_v = 0.0f;
  for (int i = lid; i < seq_k; i += local_size) {
    if (is_causal && i > group0) {
      exp_v = exp(pinput[i] * scale - max_v);
    } else {
      exp_v = exp(pinput[i] * scale + (mask == NULL ? 0.0f : mask[i]) - max_v);
    }
    poutput[i] = exp_v;
    sum += exp_v;
  }
  sdata[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int offset = local_size_half; offset >= 1; offset >>= 1) {
    if (lid < offset) {
      sdata[lid] += sdata[lid + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    sdata[0] = 1.0f / sdata[0];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float sum_v = sdata[0];
  for (int i = lid; i < seq_k; i += local_size) {
    poutput[i] *= sum_v;
  }
}