add_library(conv2d_ocl SHARED conv2d.cpp)
target_link_libraries(conv2d_ocl PRIVATE OpenCL::OpenCL)

add_executable(conv2d_test test.cpp)
target_link_libraries(conv2d_test PRIVATE conv2d_ocl OpenCL::OpenCL GTest::GTest GTest::Main)