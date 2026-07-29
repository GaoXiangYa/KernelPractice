#include "device.h"
#include "gemm_q4_1.h"

void launch_gemm_q4_1_v0(const unsigned char* A, const float* B, float* C,
                         int M, int N, int K, int block_k,
                         float alpha, float beta) {

    auto& dm = DeviceManager::get();
    auto kernel = dm.build_kernel("../src/opencl/gemm_q4_1/gemm_q4_1_v0.cl",
                                  "gemm_q4_1_v0_kernel");

    size_t blocks_per_row = K / block_k;
    size_t block_bytes    = sizeof(float) * 2 + block_k / 2;
    size_t bytes_A        = (size_t)M * blocks_per_row * block_bytes;
    size_t elems_B        = (size_t)K * N;
    size_t elems_C        = (size_t)M * N;

    auto d_A = dm.create_ro_buffer(bytes_A,           A);
    auto d_B = dm.create_ro_buffer(sizeof(float) * elems_B, B);
    auto d_C = dm.create_rw_buffer(sizeof(float) * elems_C, C);

    kernel.setArg(0, d_A);
    kernel.setArg(1, d_B);
    kernel.setArg(2, d_C);
    kernel.setArg(3, M);
    kernel.setArg(4, N);
    kernel.setArg(5, K);
    kernel.setArg(6, block_k);
    kernel.setArg(7, alpha);
    kernel.setArg(8, beta);

    // tile: BM=16 × BN=16, local = {16, 16}
    constexpr int tx = 16, ty = 16;
    int gx = ((N + tx - 1) / tx) * tx;
    int gy = ((M + ty - 1) / ty) * ty;

    cl::NDRange global(gx, gy);
    cl::NDRange local(tx, ty);

    dm.launch(kernel, global, local, "gemm_q4_1_v0_kernel");
    dm.read_buffer(d_C, sizeof(float) * elems_C, C);
}
