// 044 BatchNorm Inference: threads apply fixed per-channel statistics.
#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op044_batch_norm_inference_kernel(
    const float* x, const float* mean, const float* variance,
    const float* gamma, const float* beta, float* out, int total, int cols,
    float eps) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= total) return;
  const int channel = offset % cols;
  out[offset] = (x[offset] - mean[channel]) *
                    rsqrtf(variance[channel] + eps) * gamma[channel] +
                beta[channel];
}

}  // namespace

cudaError_t launch_op044_batch_norm_inference(
    const float* x, const float* mean, const float* variance,
    const float* gamma, const float* beta, float* out, int rows, int cols,
    float eps, cudaStream_t stream) {
  CUDA_CHECK(reductions_detail::validate_matrix(rows, cols));
  if (!std::isfinite(eps) || eps < 0.0f) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  const int total = rows * cols;
  op044_batch_norm_inference_kernel
      <<<ceil_div_int(total, kThreads), kThreads, 0, stream>>>(
          x, mean, variance, gamma, beta, out, total, cols, eps);
  return cudaGetLastError();
}
}  // namespace gpu_ops
