find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)

add_library(gemm_cpu
  STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_v0.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_v1.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_v2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_v5.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_v6.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_v7.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_v8.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x4block_v3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x4block_v4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x4block_v5.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x4block_v6.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x4block_v7.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x4block_v8.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x4block_v9.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x4block_v10.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x4block_v11.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x8block_v12.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x8block_v13.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x8block_v14.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x8block_v15.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x8block_v16.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x8block_v17.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x8block_v18.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x8block_v19.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_4x8block_v20.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_omp_v18.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gemm_blas.cpp
)

target_link_libraries(gemm_cpu PRIVATE ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(gemm_test gemm_test.cpp)
target_link_libraries(gemm_test PRIVATE gemm_cpu ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)


add_executable(gemm_benchmark gemm_benchmark.cpp)
target_link_libraries(gemm_benchmark PRIVATE gemm_cpu ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)
