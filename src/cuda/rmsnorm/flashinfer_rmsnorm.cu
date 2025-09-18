#include "flashinfer/norm.cuh"
#include "rmsnorm.h"
#include "util.h"
#include <iostream>

void flashinfer_rmsnorm(float *input, float *weight, float *output,
                        const int input_len, const float eps) {
  size_t input_size = input_len * sizeof(float);
  size_t output_size = input_len * sizeof(float);

  float *input_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&input_dev, input_size));
  CHECK_CUDA(cudaMemcpy(input_dev, input, input_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  float *weight_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&weight_dev, input_size));
  CHECK_CUDA(cudaMemcpy(weight_dev, weight, input_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  float *output_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&output_dev, output_size));
  CHECK_CUDA(cudaMemcpy(output_dev, output, output_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  // cudaStream_t stream = 0;
  flashinfer::norm::RMSNorm<float>(input_dev, weight_dev, output_dev, 1,
                                   input_len, 0, 0, eps, false);
  // cudaStreamSynchronize(stream);

  CHECK_CUDA(cudaMemcpy(output, output_dev, output_size,
                        cudaMemcpyKind::cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
}

void flashinfer_rmsnorm_benchmark() {
  std::cout << "flashinfer_rmsnorm_benchmark begin: \n";
  const int input_len = 32 * 1024 * 1024;
  std::vector<float> input(input_len, 0.0f);
  std::vector<float> output(input_len, -1.0f);
  std::vector<float> weight(input_len, 1.0f);

  const float eps = 1.0000f;
  size_t input_size = input_len * sizeof(float);
  size_t output_size = input_len * sizeof(float);

  float *input_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&input_dev, input_size));
  CHECK_CUDA(cudaMemcpy(input_dev, input.data(), input_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  float *weight_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&weight_dev, input_size));
  CHECK_CUDA(cudaMemcpy(weight_dev, weight.data(), input_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  float *output_dev = nullptr;
  CHECK_CUDA(cudaMalloc(&output_dev, output_size));
  CHECK_CUDA(cudaMemcpy(output_dev, output.data(), output_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice));

  double flops = 5 * input_len;
  double bytes = 12 * input_size;
  const int repeat = 100;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    flashinfer::norm::RMSNorm<float>(input_dev, weight_dev, output_dev, 1,
                                     input_len, 0, 0, eps, false);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  elapsed_ms /= repeat;  // 平均单次 ms
  elapsed_ms /= 1000.0f; // 转换成秒

  double bandwidth = bytes / elapsed_ms / 1e9;
  double gflops = flops / elapsed_ms / 1e9;

  printf("Input size: %d\n", input_len);
  printf("Avg Time: %.6f s\n", elapsed_ms);
  printf("Bandwidth: %.2f GB/s\n", bandwidth);
  printf("FLOPS: %.2f GFLOPS\n", gflops);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::cout << "flashinfer_rmsnorm_benchmark end: \n";
}