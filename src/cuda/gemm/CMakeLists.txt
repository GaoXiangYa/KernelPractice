add_library(matmul STATIC ${CMAKE_CURRENT_SOURCE_DIR}/matmul_native.cu)

add_executable(matmul_test  ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)
target_link_libraries(matmul_test PRIVATE matmul)