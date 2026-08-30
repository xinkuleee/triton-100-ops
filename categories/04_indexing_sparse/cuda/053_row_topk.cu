#include "../../../shared/cuda_common.cuh"

#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace gpu_ops {
namespace {

struct Op053ValueIndex {
  float value;
  int index;
};

__device__ __forceinline__ Op053ValueIndex op053_better(
    Op053ValueIndex left, Op053ValueIndex right) {
  return (right.value > left.value ||
          (right.value == left.value && right.index < left.index))
             ? right
             : left;
}

// Repeated block reductions are readable but O(k*C), not a radix/select kernel.
__global__ void op053_row_topk_kernel(
    const float* x, float* values, int64_t* indices, int cols, int k) {
  __shared__ Op053ValueIndex candidates[kThreads];
  __shared__ int selected[64];
  const int row = blockIdx.x;
  for (int rank = 0; rank < k; ++rank) {
    Op053ValueIndex local{-CUDART_INF_F, INT_MAX};
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
      bool used = false;
      for (int prior = 0; prior < rank; ++prior)
        used |= selected[prior] == col;
      if (!used)
        local = op053_better(
            local, Op053ValueIndex{x[row * cols + col], col});
    }
    candidates[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride)
        candidates[threadIdx.x] = op053_better(
            candidates[threadIdx.x], candidates[threadIdx.x + stride]);
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      selected[rank] = candidates[0].index;
      values[row * k + rank] = candidates[0].value;
      indices[row * k + rank] = candidates[0].index;
    }
    __syncthreads();
  }
}

}  // namespace

cudaError_t launch_op053_row_topk(
    const float* x, float* values, int64_t* indices,
    int rows, int cols, int k, cudaStream_t stream) {
  if (rows < 0 || cols <= 0 || k <= 0 || k > cols || k > 64)
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  op053_row_topk_kernel<<<rows, kThreads, 0, stream>>>(
      x, values, indices, cols, k);
  return cudaGetLastError();
}

}  // namespace gpu_ops
