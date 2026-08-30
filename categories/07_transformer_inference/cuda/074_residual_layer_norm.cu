#include "_common.cuh"

namespace gpu_ops {
namespace {

// Residual add and LayerNorm share one CTA and one global write.
__global__ void op074_residual_layer_norm_kernel(
    const float* x, const float* residual, const float* gamma,
    const float* beta, float* out, int cols, float eps) {
  const int row = blockIdx.x;
  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local_sum += x[row * cols + col] + residual[row * cols + col];
  __shared__ float saved_mean;
  const float mean_value = block_sum(local_sum) / cols;
  if (threadIdx.x == 0) saved_mean = mean_value;
  __syncthreads();
  const float mean = saved_mean;
  float local_square = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = x[row * cols + col] + residual[row * cols + col] - mean;
    local_square += centered * centered;
  }
  const float inverse = rsqrtf(block_sum(local_square) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = x[row * cols + col] + residual[row * cols + col] - mean;
    out[row * cols + col] = centered * inverse * gamma[col] + beta[col];
  }
}

}  // namespace

cudaError_t launch_op074_residual_layer_norm(
    const float* x, const float* residual, const float* gamma,
    const float* beta, float* out, int rows, int cols, float eps,
    cudaStream_t stream) {
  if (rows < 0 || cols <= 0 || !std::isfinite(eps) || eps <= 0.0f)
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  op074_residual_layer_norm_kernel<<<
      rows, kThreads, 0, stream>>>(
      x, residual, gamma, beta, out, cols, eps);
  return cudaGetLastError();
}

}  // namespace gpu_ops

