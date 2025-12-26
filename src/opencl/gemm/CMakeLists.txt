add_library(gemm_ocl SHARED gemm.cpp)
target_link_libraries(gemm_ocl PRIVATE OpenCL::OpenCL)

add_executable(gemm_test test.cpp)
target_link_libraries(gemm_test PRIVATE gemm_ocl OpenCL::OpenCL GTest::GTest GTest::Main)