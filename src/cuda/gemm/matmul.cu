#include "matmul.cuh"
#include <cstdio>
#include <cuda_runtime.h>

// a[m, n] * b[n, k] = c[m, k]
__global__ void matmul_native_kernel(const float *a, const float *b, float *c,
                                     int m, int n, int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < k) {
    float sum = 0;
    for (int i = 0; i < n; ++i) {
      sum += a[row * n + i] * b[i * k + col];
    }
    c[row * k + col] = sum;
  }
}

// 使用shared memory进行优化
// a[m, n] * b[n, k] = c[m, k]
template <int BLOCK>
__global__ void matmul_sharedmemory_kernel(const float *__restrict__ matrix_a,
                                           const float *__restrict__ matrix_b,
                                           float *__restrict__ matrix_c, int m,
                                           int n, int k) {
  // dynamic allocate shared memory
  extern __shared__ float shared_mem[];
  float *shmem_a = shared_mem;
  float *shmem_b = &shared_mem[BLOCK * BLOCK];

  const int col = blockIdx.x * BLOCK + threadIdx.x;
  const int row = blockIdx.y * BLOCK + threadIdx.y;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;

  float sum = 0.0f;
  for (int ph = 0; ph < (n + BLOCK - 1) / BLOCK; ++ph) {

    int a_col = ph * BLOCK + tx;
    int a_idx = row * n + a_col;

    int b_row = ph * BLOCK + ty;
    int b_idx = b_row * k + col;

    if (row < m && a_col < n) {
      shmem_a[ty * BLOCK + tx] = matrix_a[a_idx];
    } else {
      shmem_a[ty * BLOCK + tx] = 0.0f;
    }

    if (b_row < n && col < k) {
      shmem_b[ty * BLOCK + tx] = matrix_b[b_idx];
    } else {
      shmem_b[ty * BLOCK + tx] = 0.0f;
    }

    __syncthreads();

    for (int i = 0; i < BLOCK; ++i) {
      sum += shmem_a[ty * BLOCK + i] * shmem_b[i * BLOCK + tx];
    }

    __syncthreads();
  }

  if (row < m && col < k) {
    matrix_c[row * k + col] = sum;
  }
}

template <int BLOCK_SIZE, int COARSE_FACTOR>
__global__ void matmul_sharedmemory_threadcoarsening_kernel(
    const float *__restrict__ matrix_a, const float *__restrict__ matrix_b,
    float *__restrict__ matrix_c, int m, int n, int k) {
  extern __shared__ float shared_mem[];
  float *shmem_a = shared_mem;
  float *shmem_b = &shared_mem[BLOCK_SIZE * BLOCK_SIZE * COARSE_FACTOR];

  int col_start = blockIdx.x * BLOCK_SIZE * COARSE_FACTOR + threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int tile_row = threadIdx.y;
  int tile_col = threadIdx.x;

  float sum[COARSE_FACTOR];

#pragma unroll
  for (int c = 0; c < COARSE_FACTOR; ++c) {
    sum[c] = 0.0f;
  }

  for (int ph = 0; ph < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++ph) {
    int a_col = ph * BLOCK_SIZE + tile_col;
    int a_idx = row * n + a_col;

    if (row < m && a_col < n) {
      shmem_a[tile_row * BLOCK_SIZE + tile_col] = matrix_a[a_idx];
    } else {
      shmem_a[tile_row * BLOCK_SIZE + tile_col] = 0.0f;
    }

#pragma unroll
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      int col = col_start + c * BLOCK_SIZE + tile_col;
      int b_row = ph * BLOCK_SIZE + tile_row;
      int b_idx = b_row * k + col;

      if (b_row < n && col < k) {
        shmem_b[c * BLOCK_SIZE * BLOCK_SIZE + tile_row * BLOCK_SIZE +
                tile_col] = matrix_b[b_idx];
      } else {
        shmem_b[c * BLOCK_SIZE * BLOCK_SIZE + tile_row * BLOCK_SIZE +
                tile_col] = 0.0f;
      }
    }

    __syncthreads();
#pragma unroll
    for (int c = 0; c < COARSE_FACTOR; ++c) {
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE; ++i) {
        sum[c] +=
            shmem_a[tile_row * BLOCK_SIZE + i] *
            shmem_b[c * BLOCK_SIZE * BLOCK_SIZE + i * BLOCK_SIZE + tile_col];
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int c = 0; c < COARSE_FACTOR; ++c) {
    int col = col_start + c * BLOCK_SIZE;
    if (row < m && col < k) {
      matrix_c[row * k + col] = sum[c];
    }
  }
}

void matmul_native(const float *a, const float *b, float *c, int m, int n,
                   int k) {
  float *dev_a = nullptr;
  auto err = cudaMalloc(&dev_a, m * n * sizeof(float));
  err = cudaMemcpy(dev_a, a, m * n * sizeof(float), cudaMemcpyHostToDevice);

  float *dev_b = nullptr;
  err = cudaMalloc(&dev_b, n * k * sizeof(float));
  err = cudaMemcpy(dev_b, b, n * k * sizeof(float), cudaMemcpyHostToDevice);

  float *dev_c = nullptr;
  err = cudaMalloc(&dev_c, m * k * sizeof(float));
  err = cudaMemcpy(dev_c, c, m * k * sizeof(float), cudaMemcpyHostToDevice);

  const int THREAD_COUNT = 32;
  dim3 block(THREAD_COUNT, THREAD_COUNT);
  dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y);

  matmul_native_kernel<<<grid, block>>>(dev_a, dev_b, dev_c, m, n, k);

  cudaMemcpy(c, dev_c, m * k * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}

void matmul_sharedmemory(const float *matrix_a, const float *matrix_b,
                         float *matrix_c, int m, int n, int k) {
  float *dev_a = nullptr;
  auto err = cudaMalloc(&dev_a, m * n * sizeof(float));
  err = cudaMemcpy(dev_a, matrix_a, m * n * sizeof(float),
                   cudaMemcpyHostToDevice);

  float *dev_b = nullptr;
  err = cudaMalloc(&dev_b, n * k * sizeof(float));
  err = cudaMemcpy(dev_b, matrix_b, n * k * sizeof(float),
                   cudaMemcpyHostToDevice);

  float *dev_c = nullptr;
  err = cudaMalloc(&dev_c, m * k * sizeof(float));
  err = cudaMemcpy(dev_c, matrix_c, m * k * sizeof(float),
                   cudaMemcpyHostToDevice);

  const int BLOCK_SIZE = 32;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y);

  // calculate shared memory size
  auto shared_mem_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);

  if (BLOCK_SIZE == 16) {
    matmul_sharedmemory_kernel<16>
        <<<grid, block, shared_mem_size>>>(dev_a, dev_b, dev_c, m, n, k);
  } else if (BLOCK_SIZE == 32) {
    matmul_sharedmemory_kernel<32>
        <<<grid, block, shared_mem_size>>>(dev_a, dev_b, dev_c, m, n, k);
  } else {
    matmul_sharedmemory_kernel<8>
        <<<grid, block, shared_mem_size>>>(dev_a, dev_b, dev_c, m, n, k);
  }

  cudaMemcpy(matrix_c, dev_c, m * k * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}

void matmul_sharedmemory_threadcoarsening(const float *matrix_a,
                                          const float *matrix_b,
                                          float *matrix_c, int m, int n,
                                          int k) {
  float *dev_a = nullptr;
  auto err = cudaMalloc(&dev_a, m * n * sizeof(float));
  err = cudaMemcpy(dev_a, matrix_a, m * n * sizeof(float),
                   cudaMemcpyHostToDevice);

  float *dev_b = nullptr;
  err = cudaMalloc(&dev_b, n * k * sizeof(float));
  err = cudaMemcpy(dev_b, matrix_b, n * k * sizeof(float),
                   cudaMemcpyHostToDevice);

  float *dev_c = nullptr;
  err = cudaMalloc(&dev_c, m * k * sizeof(float));
  err = cudaMemcpy(dev_c, matrix_c, m * k * sizeof(float),
                   cudaMemcpyHostToDevice);

  const int BLOCK_SIZE = 32;
  const int COARSE_FACTOR = 4;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y);

  // calculate shared memory size
  auto shared_mem_size =
      (1 + COARSE_FACTOR) * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);

  matmul_sharedmemory_threadcoarsening_kernel<BLOCK_SIZE, COARSE_FACTOR>
      <<<grid, block, shared_mem_size>>>(dev_a, dev_b, dev_c, m, n, k);

  cudaMemcpy(matrix_c, dev_c, m * k * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}