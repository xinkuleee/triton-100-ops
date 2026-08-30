#include "_common.cuh"

namespace gpu_ops {
namespace {

// RMSNorm omits mean subtraction and beta.
__global__ void op075_residual_rms_norm_kernel(
    const float* x, const float* residual, const float* gamma, float* out,
    int cols, float eps) {
  const int row = blockIdx.x;
  float local_square = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float z = x[row * cols + col] + residual[row * cols + col];
    local_square += z * z;
  }
  const float inverse = rsqrtf(block_sum(local_square) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float z = x[row * cols + col] + residual[row * cols + col];
    out[row * cols + col] = z * inverse * gamma[col];
  }
}

}  // namespace

cudaError_t launch_op075_residual_rms_norm(
    const float* x, const float* residual, const float* gamma, float* out,
    int rows, int cols, float eps, cudaStream_t stream) {
  if (rows < 0 || cols <= 0 || !std::isfinite(eps) || eps <= 0.0f)
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  op075_residual_rms_norm_kernel<<<
      rows, kThreads, 0, stream>>>(
      x, residual, gamma, out, cols, eps);
  return cudaGetLastError();
}

}  // namespace gpu_ops

