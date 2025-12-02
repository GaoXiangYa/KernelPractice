#include "gemv.h"
#include <fstream>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>

using gemv_func = std::function<void(std::ofstream &, const int, const int)>;
static std::unordered_map<std::string, gemv_func> gemv_map = {
    {"gemv_v0", benchmark_gemv_v0},
    {"gemv_v1", benchmark_gemv_v1},
    {"gemv_v2", benchmark_gemv_v2},
    {"gemv_v3", benchmark_gemv_v3},
    {"gemv_v4", benchmark_gemv_v4},
    {"gemv_v5", benchmark_gemv_v5},
    {"gemv_v6", benchmark_gemv_v6},
    {"cutlass_gemv", benchmark_cutlass_gemv_fp32}};

void launchBenchmark(const std::string &name) {
  gemv_func func = gemv_map[name];
  std::ofstream gemv_file(name + ".txt", std::ios::out);

  for (int stride = 32; stride <= 4096; stride += 32) {
    func(gemv_file, stride, stride);
  }
}

int main(int argc, const char *argv[]) {
  std::span<const char *> arg_span(argv, argc);
  for (int i = 1; i < arg_span.size(); ++i) {
    launchBenchmark(arg_span[i]);
  }
  return 0;
}
