#include "gemv.h"

int main() {

  for (int i = 32; i <= 2048; i += 32) {
    benchmark_gemv_v0(i, i);
  }

  for (int i = 32; i <= 2048; i += 32) {
    benchmark_cutlass_gemv_fp32(i, i);
  }

  return 0;
}
