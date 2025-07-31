add_library(matmul_cuda STATIC ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cu)
target_include_directories(matmul_cuda PUBLIC ${CUDA_INCLUDE_DIRS})

add_executable(matmul_test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
target_link_libraries(matmul_test PRIVATE cusparse cublas matmul_cuda)
