// 018 Leaky ReLU: one CUDA thread applies the positive/negative branch.
#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op018_leaky_relu_kernel(
    const float* x, float slope, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) out[index] = x[index] >= 0.0f ? x[index] : slope * x[index];
}
}  // namespace

cudaError_t launch_op018_leaky_relu(
    const float* x, float slope, float* out, int n, cudaStream_t stream) {
  CUDA_CHECK(elementwise_detail::validate_n(n));
  if (n == 0) return cudaSuccess;
  op018_leaky_relu_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(
      x, slope, out, n);
  return cudaGetLastError();
}
}  // namespace gpu_ops
