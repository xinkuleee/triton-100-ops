// 024 ELU: one CUDA thread selects the linear or exponential branch.
#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op024_elu_kernel(
    const float* x, float alpha, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n)
    out[index] = x[index] > 0.0f ? x[index] : alpha * expm1f(x[index]);
}
}  // namespace

cudaError_t launch_op024_elu(
    const float* x, float alpha, float* out, int n, cudaStream_t stream) {
  CUDA_CHECK(elementwise_detail::validate_n(n));
  if (n == 0) return cudaSuccess;
  op024_elu_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(
      x, alpha, out, n);
  return cudaGetLastError();
}
}  // namespace gpu_ops
