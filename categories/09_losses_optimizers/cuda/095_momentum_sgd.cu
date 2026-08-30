#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op095_momentum_sgd_kernel(
    float* param, const float* grad, float* velocity, int count,
    float learning_rate, float momentum) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= count) return;
  const float next_velocity = momentum * velocity[offset] + grad[offset];
  velocity[offset] = next_velocity;
  param[offset] -= learning_rate * next_velocity;
}

}  // namespace

cudaError_t launch_op095_momentum_sgd(
    float* param, const float* grad, float* velocity, int count,
    float learning_rate, float momentum, cudaStream_t stream) {
  if (count < 0 || !std::isfinite(learning_rate) ||
      !std::isfinite(momentum) || learning_rate < 0.0f || momentum < 0.0f)
    return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  op095_momentum_sgd_kernel<<<
      ceil_div_int(count, kThreads), kThreads, 0, stream>>>(
      param, grad, velocity, count, learning_rate, momentum);
  return cudaGetLastError();
}

}  // namespace gpu_ops
