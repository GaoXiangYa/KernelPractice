#include "device.h"
#include "rmsnorm.h"

#include <chrono>
#include <format>
#include <iostream>
#include <string>
#include <vector>

using RmsnormFunc = void (*)(const std::vector<float>&, const std::vector<float>&,
                             std::vector<float>&, int, int, float);

struct Variant {
    const char*  name;
    RmsnormFunc  func;
    const char*  cl_file;
    const char*  kernel_name;
    int          local_size;
    int          args;        // argc for setArg (5 or 6)
};

static const Variant kVariants[] = {
    {"v0", launch_rmsnorm_v0, "rmsnorm_v0.cl", "rms_norm_fused_v0", 64, 6},
    {"v1", launch_rmsnorm_v1, "rmsnorm_v1.cl", "rms_norm_fused_v1",  64, 5},
};

static void bench_one_raw(const Variant& v, const char* label,
                          int N, int D, float epsilon,
                          int warmup, int timed) {

    auto& dm = DeviceManager::get();
    auto kernel = dm.build_kernel("../src/opencl/rmsnorm/" + std::string(v.cl_file),
                                  v.kernel_name);

    size_t elems = (size_t)N * D;
    std::vector<float> input(elems);
    std::vector<float> weight(D, 1.0f);
    std::vector<float> output(elems, -1.0f);

    for (size_t i = 0; i < elems; ++i)
        input[i] = (float)(rand()) / (float)RAND_MAX * 2.0f - 1.0f;

    auto d_i = dm.create_ro_buffer(sizeof(float) * elems, input.data());
    auto d_w = dm.create_ro_buffer(sizeof(float) * D,     weight.data());
    auto d_o = dm.create_rw_buffer(sizeof(float) * elems, output.data());

    kernel.setArg(0, d_i);
    kernel.setArg(1, d_w);
    kernel.setArg(2, d_o);
    kernel.setArg(3, D);
    kernel.setArg(4, epsilon);
    if (v.args >= 6)
        kernel.setArg(5, cl::Local(v.local_size * sizeof(float)));

    cl::NDRange global(v.local_size * N);
    cl::NDRange local(v.local_size);

    // warmup
    for (int i = 0; i < warmup; ++i)
        dm.launch_silent(kernel, global, local);
    dm.queue().finish();

    // timed
    cl::Event evt_first, evt_last;
    dm.queue().enqueueNDRangeKernel(kernel, cl::NullRange, global, local,
                                    nullptr, &evt_first);
    for (int i = 1; i < timed; ++i)
        dm.launch_silent(kernel, global, local);
    dm.queue().enqueueNDRangeKernel(kernel, cl::NullRange, global, local,
                                    nullptr, &evt_last);
    evt_last.wait();

    auto t_start = evt_first.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    auto t_end   = evt_last.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double us    = (t_end - t_start) * 1e-9 / timed * 1e6;

    double flops  = 3.0 * (double)N * (double)D;
    double gflops = flops / (us * 1e-6) / 1e9;
    double bytes  = sizeof(float) * (2.0 * N * D + (double)D);
    double bw     = bytes / (us * 1e-6) / 1e9;

    std::cout << std::format(
        "{} | N={:6d} D={:5d} | {:8.1f} µs | {:7.1f} GFLOPS | {:6.1f} GB/s\n",
        label, N, D, us, gflops, bw);
}

int main() {
    constexpr float epsilon = 1e-5f;
    constexpr int   warmup  = 3;
    constexpr int   timed   = 100;

    std::cout << std::format("rmsnorm  benchmark  (warmup={}, iters={})\n\n",
                             warmup, timed);
    std::cout << "variant         N      D        µs     GFLOPS    GB/s\n";
    std::cout << "----------  ------  -----  --------  -------  ------\n";

    struct { int N, D; const char* tag; } probs[] = {
        {256,  256,  "small"},
        {512,  256,  "small"},
        {512,  4096, "med "},
        {1024, 4096, "med "},
        {2048, 4096, "med "},
        {256,  8192, "large"},
        {512,  8192, "large"},
        {128, 16384, "xl   "},
        {256, 16384, "xl   "},
    };

    for (auto& p : probs) {
        for (auto& v : kVariants) {
            std::string label = std::format("{}_{}", v.name, p.tag);
            bench_one_raw(v, label.c_str(), p.N, p.D, epsilon, warmup, timed);
        }
    }

    std::cout << std::endl;
    return 0;
}
