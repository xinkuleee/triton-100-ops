// 020 Tanh: one CUDA thread evaluates tanhf for one FP32 element.
#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op020_tanh_kernel(const float* x, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) out[index] = tanhf(x[index]);
}
}  // namespace

cudaError_t launch_op020_tanh(
    const float* x, float* out, int n, cudaStream_t stream) {
  CUDA_CHECK(elementwise_detail::validate_n(n));
  if (n == 0) return cudaSuccess;
  op020_tanh_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(x, out, n);
  return cudaGetLastError();
}
}  // namespace gpu_ops
