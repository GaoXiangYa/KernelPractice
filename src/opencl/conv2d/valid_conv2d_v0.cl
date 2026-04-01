// native valid conv2d
__kernel void valid_conv2d_v0_kernel(__global const float* input, __global const float* filter,
                               __global float* output, int input_rows, int input_cols,
                               int filter_rows, int filter_cols) {
  int out_row = get_global_id(1);
  int out_col = get_global_id(0);

  float sum = 0.0f;
  for (int r = 0; r < filter_rows; ++ r) {
    for (int c = 0; c < filter_cols; ++ c) {
      int input_r = out_row + r;
      int input_c = out_col + c;

      float val = input[input_r * input_cols + input_c];
      float w = filter[r * filter_cols + c];

      sum += val * w;
    }
  }

  output[out_row * get_global_size(0) + out_col] = sum;
}