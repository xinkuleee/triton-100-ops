#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op089_mse_kernel(
    const float* prediction, const float* target, float* out, int cols) {
  const int row = blockIdx.x;
  float local = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float difference =
        prediction[row * cols + col] - target[row * cols + col];
    local += difference * difference;
  }
  const float total = block_sum(local);
  if (threadIdx.x == 0) out[row] = total / cols;
}

}  // namespace

cudaError_t launch_op089_mse(
    const float* prediction, const float* target, float* out, int rows,
    int cols, cudaStream_t stream) {
  if (rows < 0 || cols <= 0) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  op089_mse_kernel<<<rows, kThreads, 0, stream>>>(
      prediction, target, out, cols);
  return cudaGetLastError();
}

}  // namespace gpu_ops
