add_library(reduce_ocl SHARED reduce.cpp)
target_link_libraries(reduce_ocl PRIVATE OpenCL::OpenCL)

add_executable(reduce_test test.cpp)
target_link_libraries(reduce_test PRIVATE reduce_ocl OpenCL::OpenCL GTest::GTest GTest::Main)