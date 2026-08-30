#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op098_adagrad_kernel(
    float* param, const float* grad, float* state, int count,
    float learning_rate, float eps) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= count) return;
  const float gradient = grad[offset];
  const float next_state = state[offset] + gradient * gradient;
  state[offset] = next_state;
  param[offset] -= learning_rate * gradient / (sqrtf(next_state) + eps);
}

}  // namespace

cudaError_t launch_op098_adagrad(
    float* param, const float* grad, float* state, int count,
    float learning_rate, float eps, cudaStream_t stream) {
  if (count < 0 || !std::isfinite(learning_rate) || !std::isfinite(eps) ||
      learning_rate < 0.0f || eps <= 0.0f)
    return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  op098_adagrad_kernel<<<ceil_div_int(count, kThreads), kThreads, 0, stream>>>(
      param, grad, state, count, learning_rate, eps);
  return cudaGetLastError();
}

}  // namespace gpu_ops
