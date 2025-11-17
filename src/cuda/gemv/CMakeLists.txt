add_library(gemv_cuda STATIC ${CMAKE_CURRENT_SOURCE_DIR}/gemv_v0.cu)
target_include_directories(gemv_cuda PUBLIC ${CUDA_INCLUDE_DIRS})

add_executable(gemv_test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
target_link_libraries(gemv_test PRIVATE cusparse cublas matmul_cuda GTest::gtest_main)
