include_directories(${CMAKE_CURRENT_SOURCE_DIR}/benchmark)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
include_directories(/home/gx/Projects/OpenSources/flashinfer/include/)

add_subdirectory(gemm)
add_subdirectory(reduce)
add_subdirectory(rmsnorm)