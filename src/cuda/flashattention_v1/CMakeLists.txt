add_library(flashattention_v1_cuda STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/flash_attn_v0.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/flash_attn_v1.cu
)
target_include_directories(flashattention_v1_cuda PUBLIC ${CUDA_INCLUDE_DIRS})

if(USE_TORCH)
  add_executable(flashattention_v1_test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
  target_link_libraries(flashattention_v1_test PRIVATE flashattention_v1_cuda
    GTest::gtest_main
    /opt/libtorch/lib/libtorch_cpu.so
    /opt/libtorch/lib/libc10.so
  )
  target_include_directories(flashattention_v1_test PRIVATE ${TORCH_INCLUDE_DIRS})
endif()

add_executable(flashattention_v1_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cu)
target_link_libraries(flashattention_v1_benchmark PRIVATE flashattention_v1_cuda)
