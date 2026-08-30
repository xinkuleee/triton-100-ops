// 032 Where: one CUDA thread selects x or y from one byte condition.
#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op032_where_kernel(
    const unsigned char* condition, const float* x, const float* y,
    float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) out[index] = condition[index] != 0 ? x[index] : y[index];
}
}  // namespace

cudaError_t launch_op032_where(
    const unsigned char* condition, const float* x, const float* y, float* out,
    int n, cudaStream_t stream) {
  CUDA_CHECK(elementwise_detail::validate_n(n));
  if (n == 0) return cudaSuccess;
  op032_where_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(
      condition, x, y, out, n);
  return cudaGetLastError();
}
}  // namespace gpu_ops
