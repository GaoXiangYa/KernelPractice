#pragma once

template <int BM,  // Block M tile
          int BN,  // Block N tile
          int WM,  // Warp M tile
          int WN,  // Warp N tile
          int TM,  // Thread M tile
          int TN   // Thread N tile
          >
struct GemmConfig {
  static constexpr int BLOCK_TILE_M = BM;
  static constexpr int BLOCK_TILE_N = BN;
  static constexpr int BLOCK_TILE_K = 32;

  static constexpr int WARP_TILE_M = WM;
  static constexpr int WARP_TILE_N = WN;

  static constexpr int THREAD_TILE_M = TM;
  static constexpr int THREAD_TILE_N = TN;

  static constexpr int WARPS_M = BM / WM;
  static constexpr int WARPS_N = BN / WN;

  static constexpr int THREADS_M = WM / TM;
  static constexpr int THREADS_N = WN / TN;

  static_assert(THREADS_M * THREADS_N == 32,
                "Warp tile must be mappeed to 32 threads");

  static constexpr int WARPS = WARPS_M * WARPS_N;

  static constexpr int THREADS = WARPS * 32;

  static constexpr int SA_STRIDE = BM + 4;
};

template <class Config>
struct WarpTile {
  static constexpr int WM = Config::WARP_TILE_M;

  static constexpr int WN = Config::WARP_TILE_N;

  __device__ WarpTile();

  int warp_id;

  int lane_id;

  int warp_m;

  int warp_n;

  int block_row;
  int block_col;

  int global_row;
  int global_col;
};

template <class Config>
__device__ WarpTile<Config>::WarpTile() {
  int tid = threadIdx.x;

  warp_id = tid >> 5;

  lane_id = tid & 31;

  warp_m = warp_id / Config::WARPS_N;

  warp_n = warp_id % Config::WARPS_N;

  block_row = warp_m * Config::WARP_TILE_M;
  block_col = warp_n * Config::WARP_TILE_N;

  global_row = blockIdx.y * Config::BLOCK_TILE_M + warp_m * Config::WARP_TILE_M;

  global_col = blockIdx.x * Config::BLOCK_TILE_N + warp_n * Config::WARP_TILE_N;
}

template <class Config>
struct ThreadTile {
  int block_row;
  int block_col;

  int global_row;
  int global_col;

  int thread_m;

  int thread_n;
};

template <class Config>
__device__ ThreadTile<Config>
make_thread_tile(const WarpTile<Config>& warp_tile) {
  ThreadTile<Config> thread_tile;
  int lane = warp_tile.lane_id;
  thread_tile.thread_m = lane / Config::THREADS_N;
  thread_tile.thread_n = lane % Config::THREADS_N;

  thread_tile.block_row =
      warp_tile.block_row + thread_tile.thread_m * Config::THREAD_TILE_M;
  thread_tile.block_col =
      warp_tile.block_col + thread_tile.thread_n * Config::THREAD_TILE_N;

  thread_tile.global_row =
      warp_tile.global_row + thread_tile.thread_m * Config::THREAD_TILE_M;
  thread_tile.global_col =
      warp_tile.global_col + thread_tile.thread_n * Config::THREAD_TILE_N;
  return thread_tile;
}

template <typename T, int TM, int TN>
struct RegisterTile {
  T data[TM][TN];

  __device__ __forceinline__ T& operator()(int m, int n) { return data[m][n]; }
  __device__ __forceinline__ const T& operator()(int m, int n) const {
    return data[m][n];
  }

  __device__ void clear() {
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        data[i][j] = 0.0f;
      }
    }
  }
};