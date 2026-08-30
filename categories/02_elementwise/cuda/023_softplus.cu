// 023 Softplus: one CUDA thread evaluates the overflow-safe softplus formula.
#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op023_softplus_kernel(const float* x, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) out[index] = elementwise_detail::stable_softplus(x[index]);
}
}  // namespace

cudaError_t launch_op023_softplus(
    const float* x, float* out, int n, cudaStream_t stream) {
  CUDA_CHECK(elementwise_detail::validate_n(n));
  if (n == 0) return cudaSuccess;
  op023_softplus_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(x, out, n);
  return cudaGetLastError();
}
}  // namespace gpu_ops
