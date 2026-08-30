#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op093_kl_divergence_kernel(
    const float* log_p, const float* log_q, float* out, int cols) {
  const int row = blockIdx.x;
  float local = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float p = log_p[row * cols + col];
    const float q = log_q[row * cols + col];
    if (p == -CUDART_INF_F) continue;
    local += q == -CUDART_INF_F ? CUDART_INF_F : expf(p) * (p - q);
  }
  const float total = block_sum(local);
  if (threadIdx.x == 0) out[row] = total;
}

}  // namespace

cudaError_t launch_op093_kl_divergence(
    const float* log_p, const float* log_q, float* out, int rows, int cols,
    cudaStream_t stream) {
  if (rows < 0 || cols <= 0) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  op093_kl_divergence_kernel<<<rows, kThreads, 0, stream>>>(
      log_p, log_q, out, cols);
  return cudaGetLastError();
}

}  // namespace gpu_ops
