// 021 GELU: one CUDA thread evaluates the tanh approximation for one element.
#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op021_gelu_kernel(const float* x, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const float value = x[index];
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    const float inner = kSqrtTwoOverPi *
        (value + 0.044715f * value * value * value);
    out[index] = 0.5f * value * (1.0f + tanhf(inner));
  }
}
}  // namespace

cudaError_t launch_op021_gelu(
    const float* x, float* out, int n, cudaStream_t stream) {
  CUDA_CHECK(elementwise_detail::validate_n(n));
  if (n == 0) return cudaSuccess;
  op021_gelu_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(x, out, n);
  return cudaGetLastError();
}
}  // namespace gpu_ops
