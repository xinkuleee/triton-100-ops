// 022 SiLU: one CUDA thread fuses x with its sigmoid gate.
#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op022_silu_kernel(const float* x, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) out[index] = x[index] * elementwise_detail::stable_sigmoid(x[index]);
}
}  // namespace

cudaError_t launch_op022_silu(
    const float* x, float* out, int n, cudaStream_t stream) {
  CUDA_CHECK(elementwise_detail::validate_n(n));
  if (n == 0) return cudaSuccess;
  op022_silu_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(x, out, n);
  return cudaGetLastError();
}
}  // namespace gpu_ops
