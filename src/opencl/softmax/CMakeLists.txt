add_library(softmax_ocl SHARED softmax.cpp)
target_link_libraries(softmax_ocl PRIVATE OpenCL::OpenCL)

add_executable(softmax_test test.cpp)
target_link_libraries(softmax_test PRIVATE softmax_ocl OpenCL::OpenCL GTest::GTest GTest::Main)