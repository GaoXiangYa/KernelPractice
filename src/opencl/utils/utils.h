#pragma once

#include <CL/cl.h>
#include <string>

std::string read_file(const std::string &path);

inline void print_opencl_limits(cl_device_id device, cl_kernel kernel) {
  size_t max_wg_size;
  size_t max_item_sizes[3];
  cl_uint max_dims;

  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                  &max_wg_size, NULL);

  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_item_sizes),
                  max_item_sizes, NULL);

  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint),
                  &max_dims, NULL);

  printf("Device limits:\n");
  printf("  Max work-group size: %zu\n", max_wg_size);
  printf("  Max work-item sizes: [%zu, %zu, %zu]\n", max_item_sizes[0],
         max_item_sizes[1], max_item_sizes[2]);
  printf("  Max dimensions: %u\n", max_dims);

  if (kernel) {
    size_t kernel_wg_size;
    clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(size_t), &kernel_wg_size, NULL);

    printf("Kernel limits:\n");
    printf("  Max kernel work-group size: %zu\n", kernel_wg_size);
  }
}
