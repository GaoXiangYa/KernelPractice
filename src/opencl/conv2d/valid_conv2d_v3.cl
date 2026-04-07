// valid conv2d + constant memory + register blocking (each thread calculate 2x2
// output)
#define REG_X 2
#define REG_Y 2

__kernel void valid_conv2d_v3_kernel(
    __global const float* restrict input, __constant float* restrict filter,
    __global float* restrict output, const int input_rows, const int input_cols,
    const int filter_rows, const int filter_cols, const int output_rows,
    const int output_cols) {
  const int local_size_x = get_local_size(0);
  const int local_size_y = get_local_size(1);

  const int local_x = get_local_id(0);
  const int local_y = get_local_id(1);

  const int global_output_x_base = get_group_id(0) * local_size_x * REG_X;
  const int global_output_y_base = get_group_id(1) * local_size_y * REG_Y;

  float sum[REG_Y][REG_X];
#pragma unroll
  for (int reg_y = 0; reg_y < REG_Y; ++reg_y) {
#pragma unroll
    for (int reg_x = 0; reg_x < REG_X; ++reg_x) {
      sum[reg_y][reg_x] = 0.0f;
    }
  }

#pragma unroll
  for (int reg_y = 0; reg_y < REG_Y; ++reg_y) {
#pragma unroll
    for (int reg_x = 0; reg_x < REG_X; ++reg_x) {
      for (int filter_y = 0; filter_y < filter_rows; ++filter_y) {
        for (int filter_x = 0; filter_x < filter_cols; ++filter_x) {
          const int global_output_x = global_output_x_base + reg_x * local_size_x + local_x;
          const int global_output_y = global_output_y_base + reg_y * local_size_y + local_y;
          const int global_input_x = global_output_x + filter_x;
          const int global_input_y = global_output_y + filter_y;

          if (global_input_x < input_cols && global_input_y < input_rows &&
              global_output_x < output_cols && global_output_y < output_rows) {
            float val = input[global_input_y * input_cols + global_input_x];
            float f = filter[filter_y * filter_cols + filter_x];

            sum[reg_y][reg_x] += f * val;
          }
        }
      }
    }
  }

#pragma unroll
  for (int reg_y = 0; reg_y < REG_Y; ++reg_y) {
    const int global_output_y = global_output_y_base + reg_y * local_size_y + local_y;
    if (global_output_y < output_rows) {
#pragma unroll
      for (int reg_x = 0; reg_x < REG_X; ++reg_x) {
        const int global_output_x = global_output_x_base + reg_x * local_size_x + local_x;
        if (global_output_x < output_cols) {
          
          output[global_output_y * output_cols + global_output_x] =
              sum[reg_y][reg_x];
        }
      }
    }
  }
}