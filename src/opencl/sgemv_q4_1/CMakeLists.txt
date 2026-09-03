add_library(sgemv_q4_1_ocl SHARED sgemv_q4_1.cpp)
target_link_libraries(sgemv_q4_1_ocl PRIVATE OpenCL::OpenCL ocl_common)

add_executable(sgemv_q4_1_test test.cpp)
target_link_libraries(sgemv_q4_1_test PRIVATE sgemv_q4_1_ocl ocl_common
                                            OpenCL::OpenCL
                                            GTest::GTest GTest::Main)

add_executable(sgemv_q4_1_benchmark benchmark.cpp)
target_link_libraries(sgemv_q4_1_benchmark PRIVATE sgemv_q4_1_ocl ocl_common
                                                 OpenCL::OpenCL)
