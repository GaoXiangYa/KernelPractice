#pragma once

#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

inline std::string read_file(const std::string &path) {
  // 打开文件
  std::ifstream ifs(path, std::ios::in | std::ios::binary);
  if (!ifs) {
    std::cerr << "Error: Could not open file " << path << std::endl;
    return "";
  }

  // 获取文件的大小
  ifs.seekg(0, std::ios::end);
  std::streamsize size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  // 如果文件为空，直接返回空字符串
  if (size == 0) {
    return "";
  }

  // 读取文件内容
  std::string text(size, '\0'); // 分配足够的空间来存储文件内容
  if (ifs.read(&text[0], size)) {
    return text;
  } else {
    std::cerr << "Error: Failed to read the file " << path << std::endl;
    return "";
  }
}

template <typename T>
void set_random_values(std::vector<T> &input, T min, T max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(min, max);

  for (auto &num : input) {
    num = dis(gen);
  }
}

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

inline void print_kernel_profiling_info(const char *kernel_name,
                                        const cl::Event &event) {
  cl_ulong time_start, time_end;

  auto start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  auto end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  double nanoSeconds = static_cast<double>(end - start);

  std::cout << std::format("Kernel {} execution time: {} ms\n", kernel_name,
                           nanoSeconds / 1e6);
}
