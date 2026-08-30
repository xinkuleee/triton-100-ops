// 041 Argmax: a value-index reduction returns the leftmost row maximum.
#include "_common.cuh"

namespace gpu_ops {
namespace {

struct ValueIndex {
  float value;
  int index;
};

__device__ __forceinline__ ValueIndex better_max(
    ValueIndex left, ValueIndex right) {
  return (right.value > left.value ||
          (right.value == left.value && right.index < left.index))
      ? right : left;
}

__device__ __forceinline__ ValueIndex warp_argmax(ValueIndex item) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    const ValueIndex other{
        __shfl_down_sync(0xffffffffu, item.value, offset),
        __shfl_down_sync(0xffffffffu, item.index, offset)};
    item = better_max(item, other);
  }
  return item;
}

__device__ __forceinline__ ValueIndex block_argmax(ValueIndex item) {
  __shared__ ValueIndex partial[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  item = warp_argmax(item);
  if (lane == 0) partial[warp] = item;
  __syncthreads();
  item = threadIdx.x < ((blockDim.x + 31) >> 5)
      ? partial[lane] : ValueIndex{-CUDART_INF_F, INT_MAX};
  if (warp == 0) item = warp_argmax(item);
  if (threadIdx.x == 0) partial[0] = item;
  __syncthreads();
  return partial[0];
}

__global__ void op041_argmax_kernel(
    const float* x, int64_t* out, int cols) {
  ValueIndex local{-CUDART_INF_F, INT_MAX};
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local = better_max(local, ValueIndex{x[blockIdx.x * cols + col], col});
  const ValueIndex result = block_argmax(local);
  if (threadIdx.x == 0) out[blockIdx.x] = result.index;
}

}  // namespace

cudaError_t launch_op041_argmax(
    const float* x, int64_t* out, int rows, int cols, cudaStream_t stream) {
  CUDA_CHECK(reductions_detail::validate_matrix(rows, cols));
  if (rows == 0) return cudaSuccess;
  op041_argmax_kernel<<<rows, kThreads, 0, stream>>>(x, out, cols);
  return cudaGetLastError();
}
}  // namespace gpu_ops
