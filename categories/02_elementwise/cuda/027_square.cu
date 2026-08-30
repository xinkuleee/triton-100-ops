// 027 Square: one CUDA thread multiplies one FP32 element by itself.
#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op027_square_kernel(const float* x, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) out[index] = x[index] * x[index];
}
}  // namespace

cudaError_t launch_op027_square(
    const float* x, float* out, int n, cudaStream_t stream) {
  CUDA_CHECK(elementwise_detail::validate_n(n));
  if (n == 0) return cudaSuccess;
  op027_square_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(x, out, n);
  return cudaGetLastError();
}
}  // namespace gpu_ops
