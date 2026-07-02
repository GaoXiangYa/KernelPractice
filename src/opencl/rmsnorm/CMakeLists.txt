list(APPEND CMAKE_PREFIX_PATH /opt/libtorch)
find_package(Torch REQUIRED)

add_ocl_kernel(rmsnorm torch_cpu c10)
target_include_directories(rmsnorm_test PRIVATE ${TORCH_INCLUDE_DIRS})

# benchmark
add_executable(rmsnorm_benchmark benchmark.cpp)
target_link_libraries(rmsnorm_benchmark PRIVATE rmsnorm_ocl ocl_common
                                               OpenCL::OpenCL)
