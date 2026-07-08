add_ocl_kernel(gemm)

add_executable(gemm_benchmark benchmark.cpp)
target_link_libraries(gemm_benchmark PRIVATE gemm_ocl ocl_common
                                            OpenCL::OpenCL)
