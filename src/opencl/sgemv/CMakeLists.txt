add_ocl_kernel(sgemv)

add_executable(sgemv_benchmark benchmark.cpp)
target_link_libraries(sgemv_benchmark PRIVATE sgemv_ocl ocl_common
                                            OpenCL::OpenCL)
