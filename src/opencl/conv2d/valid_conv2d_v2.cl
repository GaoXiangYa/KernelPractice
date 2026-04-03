// valid conv2d + constant memory + shared memory
__kernel void valid_conv2d_v2_kernel(__global const float* restrict input,
                                     __constant float* restrict filter,
                                     __global float* restrict output,
                                     __local float* restrict shmem_input,
                                     int input_rows, int input_cols,
                                     int filter_rows, int filter_cols,
                                     int output_rows, int output_cols) {
  const int local_size_y = get_local_size(1);
  const int local_size_x = get_local_size(0);
  const int num_threads = local_size_x * local_size_y;

  const int shared_input_size_y = local_size_y + filter_rows - 1;
  const int shared_input_size_x = local_size_x + filter_cols - 1;
  const int shared_input_size = shared_input_size_x * shared_input_size_y;

  const int local_x = get_local_id(0);
  const int local_y = get_local_id(1);
  const int tid = local_y * local_size_x + local_x;

  const int global_y_base = local_size_y * get_group_id(1);
  const int global_x_base = local_size_x * get_group_id(0);

  // load input in shared memory
  for (int i = tid; i < shared_input_size; i += num_threads) {
    const int shmem_y = i / shared_input_size_x;
    const int shmem_x = i % shared_input_size_x;

    const int global_input_y = global_y_base + shmem_y;
    const int global_input_x = global_x_base + shmem_x;

    if (global_input_y < input_rows && global_input_x < input_cols) {
      shmem_input[i] = input[global_input_y * input_cols + global_input_x];
    }
  }
  // sync
  barrier(CLK_LOCAL_MEM_FENCE);

  // compute
  float sum = 0.0f;
  for (int r = 0; r < filter_rows; ++r) {
    for (int c = 0; c < filter_cols; ++c) {
      int input_y = local_y + r;
      int input_x = local_x + c;

      if (input_y < shared_input_size_y && input_x < shared_input_size_x) {
        float val = shmem_input[input_y * shared_input_size_x + input_x];
        float w = filter[r * filter_cols + c];
        sum += val * w;
      }
    }
  }

  const int global_output_x = global_x_base + local_x;
  const int global_output_y = global_y_base + local_y;
  if (global_output_y < output_rows && global_output_x < output_cols) {
    output[global_output_y * output_cols + global_output_x] = sum;
  }
}
