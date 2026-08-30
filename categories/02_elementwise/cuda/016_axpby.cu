// 016 AXPBY: one CUDA thread evaluates alpha*x + beta*y for one element.
#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op016_axpby_kernel(
    const float* x, const float* y, float alpha, float beta, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) out[index] = alpha * x[index] + beta * y[index];
}
}  // namespace

cudaError_t launch_op016_axpby(
    const float* x, const float* y, float alpha, float beta, float* out, int n,
    cudaStream_t stream) {
  CUDA_CHECK(elementwise_detail::validate_n(n));
  if (n == 0) return cudaSuccess;
  op016_axpby_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(
      x, y, alpha, beta, out, n);
  return cudaGetLastError();
}
}  // namespace gpu_ops
