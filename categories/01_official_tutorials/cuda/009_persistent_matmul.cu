#include "../../../shared/cuda_common.cuh"

#include <cuda.h>

#include <climits>

namespace gpu_ops {
namespace {

__device__ __forceinline__ void op009_grouped_tile_coordinates(
    int tile, int tiles_m, int tiles_n, int* tile_m, int* tile_n) {
  constexpr int kGroupM = 8;
  const int programs_per_group = kGroupM * tiles_n;
  const int group = tile / programs_per_group;
  const int first_m = group * kGroupM;
  const int remaining_m = tiles_m - first_m;
  const int actual_m = remaining_m < kGroupM ? remaining_m : kGroupM;
  const int local = tile % programs_per_group;
  *tile_m = first_m + local % actual_m;
  *tile_n = local / actual_m;
}

// 009: portable persistent tile scheduler. The host-side CUtensorMap encoder
// below is the real TMA descriptor API; a handwritten production TMA GEMM must
// additionally issue cp.async.bulk.tensor and manage mbarrier pipelines.
__global__ void op009_persistent_matmul_kernel(
    const float* a, const float* b, float* c, int m, int n, int k) {
  constexpr int kTile = 16;
  __shared__ float a_tile[kTile][kTile];
  __shared__ float b_tile[kTile][kTile];
  const int tiles_m = ceil_div_int(m, kTile);
  const int tiles_n = ceil_div_int(n, kTile);
  for (int tile = blockIdx.x; tile < tiles_m * tiles_n; tile += gridDim.x) {
    int tile_m = 0;
    int tile_n = 0;
    op009_grouped_tile_coordinates(
        tile, tiles_m, tiles_n, &tile_m, &tile_n);
    const int row = tile_m * kTile + threadIdx.y;
    const int col = tile_n * kTile + threadIdx.x;
    float accumulator = 0.0f;
    for (int k0 = 0; k0 < k; k0 += kTile) {
      const int a_col = k0 + threadIdx.x;
      const int b_row = k0 + threadIdx.y;
      a_tile[threadIdx.y][threadIdx.x] =
          row < m && a_col < k ? a[row * k + a_col] : 0.0f;
      b_tile[threadIdx.y][threadIdx.x] =
          b_row < k && col < n ? b[b_row * n + col] : 0.0f;
      __syncthreads();
      #pragma unroll
      for (int inner = 0; inner < kTile; ++inner)
        accumulator += a_tile[threadIdx.y][inner] *
                       b_tile[inner][threadIdx.x];
      __syncthreads();
    }
    if (row < m && col < n) c[row * n + col] = accumulator;
    __syncthreads();
  }
}

// One CTA per output tile: the non-persistent baseline from the official
// tutorial, kept separate so scheduling can be compared independently of math.
__global__ void op009_naive_matmul_kernel(
    const float* a, const float* b, float* c, int m, int n, int k) {
  constexpr int kTile = 16;
  __shared__ float a_tile[kTile][kTile];
  __shared__ float b_tile[kTile][kTile];
  const int tiles_n = ceil_div_int(n, kTile);
  const int tiles_m = ceil_div_int(m, kTile);
  int tile_m = 0;
  int tile_n = 0;
  op009_grouped_tile_coordinates(
      blockIdx.x, tiles_m, tiles_n, &tile_m, &tile_n);
  const int row = tile_m * kTile + threadIdx.y;
  const int col = tile_n * kTile + threadIdx.x;
  float accumulator = 0.0f;
  for (int k0 = 0; k0 < k; k0 += kTile) {
    const int a_col = k0 + threadIdx.x;
    const int b_row = k0 + threadIdx.y;
    a_tile[threadIdx.y][threadIdx.x] =
        row < m && a_col < k ? a[row * k + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        b_row < k && col < n ? b[b_row * n + col] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int inner = 0; inner < kTile; ++inner)
      accumulator += a_tile[threadIdx.y][inner] *
                     b_tile[inner][threadIdx.x];
    __syncthreads();
  }
  if (row < m && col < n) c[row * n + col] = accumulator;
}

}  // namespace

cudaError_t launch_op009_naive_matmul(
    const float* a, const float* b, float* c, int m, int n, int k,
    cudaStream_t stream) {
  if (m < 0 || n < 0 || k < 0) return cudaErrorInvalidValue;
  if (m == 0 || n == 0) return cudaSuccess;
  if (c == nullptr || (k != 0 && (a == nullptr || b == nullptr)))
    return cudaErrorInvalidValue;
  constexpr int kTile = 16;
  const int tiles_m = ceil_div_int(m, kTile);
  const int tiles_n = ceil_div_int(n, kTile);
  if (tiles_n != 0 && tiles_m > INT_MAX / tiles_n)
    return cudaErrorInvalidValue;
  const int blocks = tiles_m * tiles_n;
  op009_naive_matmul_kernel<<<blocks, dim3(kTile, kTile), 0, stream>>>(
      a, b, c, m, n, k);
  return cudaGetLastError();
}

cudaError_t launch_op009_persistent_matmul(
    const float* a, const float* b, float* c, int m, int n, int k,
    int workers, cudaStream_t stream) {
  if (m < 0 || n < 0 || k < 0 || workers <= 0) return cudaErrorInvalidValue;
  if (m == 0 || n == 0) return cudaSuccess;
  if (c == nullptr || (k != 0 && (a == nullptr || b == nullptr)))
    return cudaErrorInvalidValue;
  constexpr int kTile = 16;
  const int tiles_m = ceil_div_int(m, kTile);
  const int tiles_n = ceil_div_int(n, kTile);
  if (tiles_n != 0 && tiles_m > INT_MAX / tiles_n)
    return cudaErrorInvalidValue;
  const int total_tiles = tiles_m * tiles_n;
  const int active_workers = workers < total_tiles ? workers : total_tiles;
  op009_persistent_matmul_kernel<<<
      active_workers, dim3(16, 16), 0, stream>>>(a, b, c, m, n, k);
  return cudaGetLastError();
}

#if CUDA_VERSION >= 12000
// Real CUDA Driver API example used before issuing Hopper TMA instructions.
// Dimensions are expressed innermost-first, as required by CUtensorMap.
CUresult op009_encode_tma_tensor_map_2d(
    CUtensorMap* map, void* base, uint64_t rows, uint64_t cols,
    uint32_t tile_rows, uint32_t tile_cols) {
  if (map == nullptr || base == nullptr || rows == 0 || cols == 0 ||
      tile_rows == 0 || tile_cols == 0 || tile_rows > rows ||
      tile_cols > cols || cols > UINT64_MAX / sizeof(float))
    return CUDA_ERROR_INVALID_VALUE;
  const cuuint64_t global_dim[2] = {cols, rows};
  const cuuint64_t global_stride[1] = {cols * sizeof(float)};
  const cuuint32_t box_dim[2] = {tile_cols, tile_rows};
  const cuuint32_t element_stride[2] = {1, 1};
  return cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, base, global_dim,
      global_stride, box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
#endif

}  // namespace gpu_ops
