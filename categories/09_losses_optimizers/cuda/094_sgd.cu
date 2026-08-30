#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op094_sgd_kernel(
    float* param, const float* grad, int count, float learning_rate) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < count) param[offset] -= learning_rate * grad[offset];
}

}  // namespace

cudaError_t launch_op094_sgd(
    float* param, const float* grad, int count, float learning_rate,
    cudaStream_t stream) {
  if (count < 0 || !std::isfinite(learning_rate) || learning_rate < 0.0f)
    return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  op094_sgd_kernel<<<ceil_div_int(count, kThreads), kThreads, 0, stream>>>(
      param, grad, count, learning_rate);
  return cudaGetLastError();
}

}  // namespace gpu_ops
