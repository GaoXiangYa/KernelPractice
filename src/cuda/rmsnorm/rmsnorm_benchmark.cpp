#include "rmsnorm.h"

int main() {
  {
    rmsnorm_v0_benchmark();
  }

  {
    flashinfer_rmsnorm_benchmark();
  }

  {
    rmsnorm_v1_benchmark();
  }

  return 0;
}