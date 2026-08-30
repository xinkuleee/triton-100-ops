#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op084_per_row_quantize_kernel(
    const float* x, signed char* q, float* scales, int cols) {
  const int row = blockIdx.x;
  float local_maximum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local_maximum = fmaxf(local_maximum, fabsf(x[row * cols + col]));
  const float scale =
      quantization_detail::op082_symmetric_scale(block_max(local_maximum));
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    q[row * cols + col] =
        quantization_detail::op082_quantize_value(x[row * cols + col], scale);
  if (threadIdx.x == 0) scales[row] = scale;
}
}  // namespace

cudaError_t launch_op084_per_row_quantize(
    const float* x, signed char* q, float* scales, int rows, int cols,
    cudaStream_t stream) {
  if (rows < 0 || cols <= 0) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  op084_per_row_quantize_kernel<<<rows, kThreads, 0, stream>>>(
      x, q, scales, cols);
  return cudaGetLastError();
}
}  // namespace gpu_ops
