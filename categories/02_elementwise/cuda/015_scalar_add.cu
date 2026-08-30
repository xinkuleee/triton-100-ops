// 015 Scalar Add: every CUDA thread adds the same scalar to one FP32 element.
#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op015_scalar_add_kernel(
    const float* x, float scalar, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) out[index] = x[index] + scalar;
}
}  // namespace

cudaError_t launch_op015_scalar_add(
    const float* x, float scalar, float* out, int n, cudaStream_t stream) {
  CUDA_CHECK(elementwise_detail::validate_n(n));
  if (n == 0) return cudaSuccess;
  op015_scalar_add_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(
      x, scalar, out, n);
  return cudaGetLastError();
}
}  // namespace gpu_ops
