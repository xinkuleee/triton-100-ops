#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op096_adam_kernel(
    float* param, const float* grad, float* first_moment,
    float* second_moment, int count, float learning_rate, float beta1,
    float beta2, float eps, float correction1, float correction2) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= count) return;
  const float gradient = grad[offset];
  const float first = beta1 * first_moment[offset] +
                      (1.0f - beta1) * gradient;
  const float second = beta2 * second_moment[offset] +
                       (1.0f - beta2) * gradient * gradient;
  first_moment[offset] = first;
  second_moment[offset] = second;
  const float update = (first / correction1) /
                       (sqrtf(second / correction2) + eps);
  param[offset] -= learning_rate * update;
}

}  // namespace

cudaError_t launch_op096_adam(
    float* param, const float* grad, float* first_moment,
    float* second_moment, int count, float learning_rate, float beta1,
    float beta2, float eps, float correction1, float correction2,
    cudaStream_t stream) {
  if (count < 0 || !std::isfinite(learning_rate) || !std::isfinite(beta1) ||
      !std::isfinite(beta2) || !std::isfinite(eps) ||
      !std::isfinite(correction1) || !std::isfinite(correction2) ||
      learning_rate < 0.0f || beta1 < 0.0f || beta1 >= 1.0f ||
      beta2 < 0.0f || beta2 >= 1.0f || eps <= 0.0f || correction1 <= 0.0f ||
      correction2 <= 0.0f)
    return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  op096_adam_kernel<<<ceil_div_int(count, kThreads), kThreads, 0, stream>>>(
      param, grad, first_moment, second_moment, count, learning_rate, beta1,
      beta2, eps, correction1, correction2);
  return cudaGetLastError();
}

}  // namespace gpu_ops
