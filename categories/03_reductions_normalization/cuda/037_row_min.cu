// 037 Row Min: warp shuffles and shared memory reduce one FP32 row.
#include "_common.cuh"

namespace gpu_ops {
namespace {

__device__ __forceinline__ float warp_min_local(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fminf(value, __shfl_down_sync(0xffffffffu, value, offset));
  return value;
}

__device__ __forceinline__ float block_min_local(float value) {
  __shared__ float partial[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_min_local(value);
  if (lane == 0) partial[warp] = value;
  __syncthreads();
  float result = threadIdx.x < ((blockDim.x + 31) >> 5)
      ? partial[lane] : CUDART_INF_F;
  if (warp == 0) result = warp_min_local(result);
  if (threadIdx.x == 0) partial[0] = result;
  __syncthreads();
  return partial[0];
}

__global__ void op037_row_min_kernel(const float* x, float* out, int cols) {
  float local = CUDART_INF_F;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local = fminf(local, x[blockIdx.x * cols + col]);
  const float result = block_min_local(local);
  if (threadIdx.x == 0) out[blockIdx.x] = result;
}

}  // namespace

cudaError_t launch_op037_row_min(
    const float* x, float* out, int rows, int cols, cudaStream_t stream) {
  CUDA_CHECK(reductions_detail::validate_matrix(rows, cols));
  if (rows == 0) return cudaSuccess;
  op037_row_min_kernel<<<rows, kThreads, 0, stream>>>(x, out, cols);
  return cudaGetLastError();
}
}  // namespace gpu_ops
