// 043 RMSNorm: one CTA computes inverse RMS and applies the row scale.
#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op043_rms_norm_kernel(
    const float* x, const float* gamma, float* out, int cols, float eps) {
  __shared__ float inverse_rms;
  float local = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = x[blockIdx.x * cols + col];
    local += value * value;
  }
  const float square_sum = block_sum(local);
  if (threadIdx.x == 0) inverse_rms = rsqrtf(square_sum / cols + eps);
  __syncthreads();
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int offset = blockIdx.x * cols + col;
    out[offset] = x[offset] * inverse_rms * gamma[col];
  }
}

}  // namespace

cudaError_t launch_op043_rms_norm(
    const float* x, const float* gamma, float* out, int rows, int cols,
    float eps, cudaStream_t stream) {
  CUDA_CHECK(reductions_detail::validate_matrix(rows, cols));
  if (!std::isfinite(eps) || eps < 0.0f) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  op043_rms_norm_kernel<<<rows, kThreads, 0, stream>>>(
      x, gamma, out, cols, eps);
  return cudaGetLastError();
}
}  // namespace gpu_ops
