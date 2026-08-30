// 033 Row Bias Add: one CUDA thread adds the bias selected by its column.
#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op033_row_bias_add_kernel(
    const float* x, const float* bias, float* out, int rows, int cols) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = rows * cols;
  if (index < n) out[index] = x[index] + bias[index % cols];
}
}  // namespace

cudaError_t launch_op033_row_bias_add(
    const float* x, const float* bias, float* out, int rows, int cols,
    cudaStream_t stream) {
  if (rows < 0 || cols < 0 || (cols != 0 && rows > INT_MAX / cols))
    return cudaErrorInvalidValue;
  if (rows == 0 || cols == 0) return cudaSuccess;
  const int n = rows * cols;
  op033_row_bias_add_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(
      x, bias, out, rows, cols);
  return cudaGetLastError();
}
}  // namespace gpu_ops
