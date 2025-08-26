#include "reduce.h"
#include "test_util.h"
#include <gtest/gtest.h>
#include <vector>

TEST(recude, reduce_v0) {
  const int size = 4096;
  std::vector<float> input(size, 0.0f);
  init_random(input, 1.0f, 1.0f);

  float cuda_output;
  float cpu_output;
  reduce_v0(input.data(), size, &cuda_output);
  cpu_output = reduce_ref(input);

  const float tolerance = 0.0001f;

  std::cout << cpu_output << " " << cuda_output << "\n";
  ASSERT_LE(std::abs(cpu_output - cuda_output), tolerance);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
  return 0;
}