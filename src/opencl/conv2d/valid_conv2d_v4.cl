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

  const int group_x = get_group_id(0);
  const int group_y = get_group_id(1);

  const int global_output_x_base = group_x * local_size_x * REG_X;
  const int global_output_y_base = group_y * local_size_y * REG_Y;

  const int shared_input_size_x = local_size_x * REG_X + filter_cols - 1;
  const int shared_input_size_y = local_size_y * REG_Y + filter_rows - 1;

  const int shared_input_x_base = global_output_x_base;
  const int shared_input_y_base = global_output_y_base;

  // =========================
  // 1. load shared memory
  // =========================
  for (int y = local_y; y < shared_input_size_y; y += local_size_y) {
    int input_y = shared_input_y_base + y;

    for (int x = local_x; x < shared_input_size_x; x += local_size_x) {
      int input_x = shared_input_x_base + x;

      if (input_y < input_rows && input_x < input_cols) {
        shared_input[y * shared_input_size_x + x] =
            input[input_y * input_cols + input_x];
      } else {
        shared_input[y * shared_input_size_x + x] = 0.0f;
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // =========================
  // 2. register accumulation
  // =========================
  float sum[REG_Y][REG_X];

#pragma unroll
  for (int ry = 0; ry < REG_Y; ++ry) {
#pragma unroll
    for (int rx = 0; rx < REG_X; ++rx) {
      sum[ry][rx] = 0.0f;
    }
  }

  // =========================
  // 3. convolution
  // =========================
#pragma unroll
  for (int ry = 0; ry < REG_Y; ++ry) {
#pragma unroll
    for (int rx = 0; rx < REG_X; ++rx) {
      int out_x = global_output_x_base + rx * local_size_x + local_x;
      int out_y = global_output_y_base + ry * local_size_y + local_y;

      if (out_x >= output_cols || out_y >= output_rows)
        continue;

      for (int fy = 0; fy < filter_rows; ++fy) {
        for (int fx = 0; fx < filter_cols; ++fx) {
          int smem_x = rx * local_size_x + local_x + fx;
          int smem_y = ry * local_size_y + local_y + fy;

          float val = shared_input[smem_y * shared_input_size_x + smem_x];

          float f = filter[fy * filter_cols + fx];

          sum[ry][rx] += f * val;
        }
      }
    }
  }

  // =========================
  // 4. write back
  // =========================
#pragma unroll
  for (int ry = 0; ry < REG_Y; ++ry) {
    int out_y = global_output_y_base + ry * local_size_y + local_y;

    if (out_y < output_rows) {
#pragma unroll
      for (int rx = 0; rx < REG_X; ++rx) {
        int out_x = global_output_x_base + rx * local_size_x + local_x;

        if (out_x < output_cols) {
          output[out_y * output_cols + out_x] = sum[ry][rx];
        }
      }
    }
  }
}