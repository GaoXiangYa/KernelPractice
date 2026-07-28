add_library(gemm_q4_1_ocl SHARED gemm_q4_1.cpp)
target_link_libraries(gemm_q4_1_ocl PRIVATE OpenCL::OpenCL ocl_common)

add_executable(gemm_q4_1_test test.cpp)
target_link_libraries(gemm_q4_1_test PRIVATE gemm_q4_1_ocl ocl_common
                                           OpenCL::OpenCL
                                           GTest::GTest GTest::Main)

add_executable(gemm_q4_1_benchmark benchmark.cpp)
target_link_libraries(gemm_q4_1_benchmark PRIVATE gemm_q4_1_ocl ocl_common
                                                OpenCL::OpenCL)
