// valid conv2d + constant memory + register blocking (each thread calculate 2x2
// output) + shared memory
#define REG_X 2
#define REG_Y 2

__kernel void valid_conv2d_v4_kernel(
    __global const float* restrict input, __constant float* restrict filter,
    __global float* restrict output, __local float* restrict shared_input,
    const int input_rows, const int input_cols, const int filter_rows,
    const int filter_cols, const int output_rows, const int output_cols) {
  const int local_size_x = get_local_size(0);
  const int local_size_y = get_local_size(1);

  const int local_x = get_local_id(0);
  const int local_y = get_local_id(1);

  const int group_x = get_local_id(0);
  const int group_y = get_local_id(1);

  const int global_output_x_base = group_x * local_size_x * REG_X;
  const int global_output_y_base = group_y * local_size_y * REG_Y;

  const int shared_input_size_x = local_size_x + filter_cols - 1;
  const int shared_input_size_y = local_size_y + filter_rows - 1;

  const int shared_input_x_base = group_x * shared_input_size_x * REG_X;
  const int shared_input_y_base = group_y * shared_input_size_y * REG_Y;

  // load input to shared memory
#pragma unroll
  for (int reg_y = 0; reg_y < REG_Y; ++reg_y) {
#pragma unroll
    for (int reg_x = 0; reg_x < REG_X; ++reg_x) {
      for (int y = local_y; y < shared_input_size_y; y += local_size_y) {
        const int shared_input_y =
            shared_input_y_base + reg_y * local_size_y + y;
        if (shared_input_y < shared_input_size_y) {
          for (int x = local_x; x < shared_input_size_x; x += local_size_x) {
            const int shared_input_x =
                shared_input_x_base + reg_x * local_size_x + x;
            if (shared_input_x < shared_input_size_x) {
              shared_input[((reg_y * REG_X + reg_x) * local_size_y + local_y) *
                               local_size_x +
                           local_x] =
                  shared_input[shared_input_y * shared_input_size_x +
                               shared_input_x];
            }
          }
        }
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

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
          const int global_output_x =
              global_output_x_base + reg_x * local_size_x + local_x;
          const int global_output_y =
              global_output_y_base + reg_y * local_size_y + local_y;
          const int shared_input_x = global_output_x + filter_x;
          const int shared_input_y = global_output_y + filter_y;

          if (shared_input_x < shared_input_size_x && shared_input_y < shared_input_size_y) {
            float val = shared_input[shared_input_y * shared_input_size_x + shared_input_x];
            float f = filter[filter_y * filter_cols + filter_x];

            sum[reg_y][reg_x] += f * val;
          }
        }
      }
    }
  }

#pragma unroll
  for (int reg_y = 0; reg_y < REG_Y; ++reg_y) {
    const int global_output_y =
        global_output_y_base + reg_y * local_size_y + local_y;
    if (global_output_y < output_rows) {
#pragma unroll
      for (int reg_x = 0; reg_x < REG_X; ++reg_x) {
        const int global_output_x =
            global_output_x_base + reg_x * local_size_x + local_x;
        if (global_output_x < output_cols) {
          output[global_output_y * output_cols + global_output_x] =
              sum[reg_y][reg_x];
        }
      }
    }
  }
}