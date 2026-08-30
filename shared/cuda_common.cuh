#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace gpu_ops {

constexpr int kThreads = 256;
__host__ __device__ constexpr int ceil_div_int(int x, int y) {
  return x / y + (x % y != 0);
}

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t error = (expr);                                                 \
    if (error != cudaSuccess) return error;                                     \
  } while (0)

__device__ __forceinline__ float warp_sum(
    float value, unsigned mask = 0xffffffffu) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(mask, value, offset);
  return value;
}

__device__ __forceinline__ float warp_max(
    float value, unsigned mask = 0xffffffffu) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_down_sync(mask, value, offset));
  return value;
}

__device__ __forceinline__ float block_sum(float value) {
  __shared__ float partial[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) partial[warp] = value;
  __syncthreads();
  float result = threadIdx.x < ((blockDim.x + 31) >> 5)
                     ? partial[lane] : 0.0f;
  if (warp == 0) result = warp_sum(result);
  if (threadIdx.x == 0) partial[0] = result;
  __syncthreads();
  const float broadcast = partial[0];
  __syncthreads();  // protect the broadcast load before this helper is reused
  return broadcast;
}

__device__ __forceinline__ float block_max(float value) {
  __shared__ float partial[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_max(value);
  if (lane == 0) partial[warp] = value;
  __syncthreads();
  float result = threadIdx.x < ((blockDim.x + 31) >> 5)
                     ? partial[lane] : -INFINITY;
  if (warp == 0) result = warp_max(result);
  if (threadIdx.x == 0) partial[0] = result;
  __syncthreads();
  const float broadcast = partial[0];
  __syncthreads();
  return broadcast;
}

}  // namespace gpu_ops
