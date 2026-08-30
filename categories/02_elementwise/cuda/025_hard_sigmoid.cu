// 025 Hard Sigmoid: one CUDA thread evaluates and clamps the affine gate.
#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op025_hard_sigmoid_kernel(const float* x, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const float value = x[index] / 6.0f + 0.5f;
    out[index] = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
  }
}
}  // namespace

cudaError_t launch_op025_hard_sigmoid(
    const float* x, float* out, int n, cudaStream_t stream) {
  CUDA_CHECK(elementwise_detail::validate_n(n));
  if (n == 0) return cudaSuccess;
  op025_hard_sigmoid_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(
      x, out, n);
  return cudaGetLastError();
}
}  // namespace gpu_ops
