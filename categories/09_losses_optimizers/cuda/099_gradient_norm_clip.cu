#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op099_gradient_norm_clip_kernel(
    float* grad, int count, float norm, float max_norm, float eps) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < count)
    grad[offset] *= fminf(1.0f, max_norm / (norm + eps));
}

}  // namespace

cudaError_t launch_op099_gradient_norm_clip(
    float* grad, int count, float norm, float max_norm, float eps,
    cudaStream_t stream) {
  if (count < 0 || !std::isfinite(norm) || !std::isfinite(max_norm) ||
      !std::isfinite(eps) || norm < 0.0f || max_norm < 0.0f || eps <= 0.0f)
    return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  op099_gradient_norm_clip_kernel<<<
      ceil_div_int(count, kThreads), kThreads, 0, stream>>>(
      grad, count, norm, max_norm, eps);
  return cudaGetLastError();
}

}  // namespace gpu_ops
