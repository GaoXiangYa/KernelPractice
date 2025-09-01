#include "reduce.h"

int main() {
  // reduce v0 benchmark
  { reduce_v0_benchmark(); }

  // reduce v1 benchmark
  { reduce_v1_benchmark(); }

  { reduce_v2_benchmark(); }

  return 0;
}