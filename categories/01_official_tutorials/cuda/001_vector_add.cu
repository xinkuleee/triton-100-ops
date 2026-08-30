#include "../../../shared/cuda_common.cuh"

namespace gpu_ops {
namespace {

// 001 -----------------------------------------------------------------------
__global__ void op001_vector_add_kernel(
    const float* x, const float* y, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) out[index] = x[index] + y[index];
}

}  // namespace

cudaError_t launch_op001_vector_add(
    const float* x, const float* y, float* out, int n, cudaStream_t stream) {
  if (n < 0) return cudaErrorInvalidValue;
  if (n == 0) return cudaSuccess;
  if (x == nullptr || y == nullptr || out == nullptr)
    return cudaErrorInvalidValue;
  op001_vector_add_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(
      x, y, out, n);
  return cudaGetLastError();
}

}  // namespace gpu_ops
