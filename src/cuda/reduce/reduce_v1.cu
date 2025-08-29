#include <cuda_runtime.h>
#include <cstdlib>


template<int SHARED_MEM_SIZE>
__global__ void reduce_kernel_v1(float *input, float *output) {
  __shared__ float shmem[SHARED_MEM_SIZE];

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = threadIdx.x;
  
  shmem[threadIdx.x] = input[tid];
  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
    if (threadIdx.x < stride) {
      shmem[i] += shmem[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[blockIdx.x] = shmem[0];
  }
}

void reduce_v1(float *input, size_t input_count, float *output) {
  size_t input_size = input_count * sizeof(float);
  const int THREAD_COUNT = 32;
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

  reduce_kernel_v1<THREAD_COUNT><<<BLOCK_COUNT, THREAD_COUNT>>>(input_dev, output_dev);

  cudaMemcpy(output_host, output_dev, output_size,
             cudaMemcpyKind::cudaMemcpyDeviceToHost);

  float sum = 0.0f;
  for (int i = 0; i < BLOCK_COUNT; ++i) {
    sum += output_host[i];
  }

  *output = sum;
}