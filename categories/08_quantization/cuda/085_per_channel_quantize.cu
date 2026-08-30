#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op085_per_channel_quantize_kernel(
    const float* x, signed char* q, float* scales, int rows, int cols) {
  const int col = blockIdx.x;
  float local_maximum = 0.0f;
  for (int row = threadIdx.x; row < rows; row += blockDim.x)
    local_maximum = fmaxf(local_maximum, fabsf(x[row * cols + col]));
  const float scale =
      quantization_detail::op082_symmetric_scale(block_max(local_maximum));
  for (int row = threadIdx.x; row < rows; row += blockDim.x)
    q[row * cols + col] =
        quantization_detail::op082_quantize_value(x[row * cols + col], scale);
  if (threadIdx.x == 0) scales[col] = scale;
}
}  // namespace

cudaError_t launch_op085_per_channel_quantize(
    const float* x, signed char* q, float* scales, int rows, int cols,
    cudaStream_t stream) {
  if (rows <= 0 || cols < 0) return cudaErrorInvalidValue;
  if (cols == 0) return cudaSuccess;
  op085_per_channel_quantize_kernel<<<cols, kThreads, 0, stream>>>(
      x, q, scales, rows, cols);
  return cudaGetLastError();
}
}  // namespace gpu_ops
