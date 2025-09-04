#include "reduce.h"

int main() {
  // reduce v0 benchmark
  { reduce_v0_benchmark(); }

  // reduce v1 benchmark
  { reduce_v1_benchmark(); }

  // reduce v2 benchmark
  { reduce_v2_benchmark(); }

  // reduce v3 benchmark
  { reduce_v3_benchmark(); }

  // reduce v4 benchmark
  { reduce_v4_benchmark(); }

  // reduce v5 benchmark
  { reduce_v5_benchmark(); }
  return 0;
}