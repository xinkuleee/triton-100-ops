// 031 Clamp: one CUDA thread restricts one element to [low, high].
#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op031_clamp_kernel(
    const float* x, float low, float high, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const float value = x[index];
    out[index] = value < low ? low : (value > high ? high : value);
  }
}
}  // namespace

cudaError_t launch_op031_clamp(
    const float* x, float low, float high, float* out, int n,
    cudaStream_t stream) {
  if (n < 0 || low > high) return cudaErrorInvalidValue;
  if (n == 0) return cudaSuccess;
  op031_clamp_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(
      x, low, high, out, n);
  return cudaGetLastError();
}
}  // namespace gpu_ops
