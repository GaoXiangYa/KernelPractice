cmake_minimum_required(VERSION 3.20)
project(KernelPractice CXX C CUDA)

option(USE_CUDA "Enable CUDA" ON)
message(STATUS "Option USE_CUDA: ${USE_CUDA}")
option(USE_OPENCL OFF)
message(STATUS "Option USE_OPENCL: ${USE_OPENCL}")
option(USE_GTEST OFF)
message(STATUS "Option USE_GTEST: ${USE_GTEST}")
option(USE_DEMO OFF)
message(STATUS "Option USE_DEMO: ${USE_DEMO}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Build type is Debug")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Werror")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls")
  message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
endif()

if(USE_CUDA)
  # 查找 CUDA
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  # enable_language(CUDA)
endif()

add_subdirectory(src/cuda)
