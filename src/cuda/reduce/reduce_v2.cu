#include <cuda_runtime.h>

template <int SHARED_MEM_SIZE, int COARSE_FACTOR>
__global__ void reduce_kernel_v2(float *input, float *output) {
  __shared__ float shmem[SHARED_MEM_SIZE];

  const int segment = 2 * COARSE_FACTOR * blockDim.x * blockIdx.x;
  const int tx = threadIdx.x;
  const int i = segment + tx;
  float sum = 0.0f;

#pragma unroll
  for (int tile = 1; tile <= COARSE_FACTOR * 2; ++tile) {
    sum += input[i + tile];
  }
  shmem[tx] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
    if (tx < stride) {
      shmem[tx] += shmem[tx + stride];
    }
    __syncthreads();
  }

  if (tx == 0) {
    output[blockIdx.x] = shmem[0];
  }
}

void reduce_v2(float *input, size_t input_count, float *output) {
  size_t input_size = input_count * sizeof(float);
  const int THREAD_COUNT = 32;
  const int COARSE_FACTOR = 2;
  const int BLOCK_COUNT = (input_count + THREAD_COUNT - 1) / THREAD_COUNT;

  size_t output_size = BLOCK_COUNT * sizeof(float);

  float *input_dev = nullptr;
  auto err = cudaMalloc(&input_dev, input_size);
  cudaMemcpy(input_dev, input, input_size,
             cudaMemcpyKind::cudaMemcpyHostToDevice);

  float *output_dev = nullptr;
  err = cudaMalloc(&output_dev, output_size);
  cudaMemcpy(output_dev, output, output_size,
             cudaMemcpyKind::cudaMemcpyHostToDevice);
  float *output_host = (float *)std::malloc(output_size);

  reduce_kernel_v2<THREAD_COUNT, COARSE_FACTOR>
      <<<BLOCK_COUNT, THREAD_COUNT>>>(input_dev, output_dev);

  cudaMemcpy(output_host, output_dev, output_size,
             cudaMemcpyKind::cudaMemcpyDeviceToHost);

  float sum = 0.0f;
  for (int i = 0; i < BLOCK_COUNT; ++i) {
    sum += output_host[i];
  }

  *output = sum;
}