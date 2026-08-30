// 014 Div: one CUDA thread divides one FP32 element pair.
#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op014_div_kernel(const float* x, const float* y, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) out[index] = x[index] / y[index];
}
}  // namespace

cudaError_t launch_op014_div(
    const float* x, const float* y, float* out, int n, cudaStream_t stream) {
  CUDA_CHECK(elementwise_detail::validate_n(n));
  if (n == 0) return cudaSuccess;
  op014_div_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(x, y, out, n);
  return cudaGetLastError();
}
}  // namespace gpu_ops
