// 028 Sqrt: one CUDA thread evaluates sqrtf for one FP32 element.
#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op028_sqrt_kernel(const float* x, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) out[index] = sqrtf(x[index]);
}
}  // namespace

cudaError_t launch_op028_sqrt(
    const float* x, float* out, int n, cudaStream_t stream) {
  CUDA_CHECK(elementwise_detail::validate_n(n));
  if (n == 0) return cudaSuccess;
  op028_sqrt_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(x, out, n);
  return cudaGetLastError();
}
}  // namespace gpu_ops
