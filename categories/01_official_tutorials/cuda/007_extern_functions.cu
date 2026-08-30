#include "../../../shared/cuda_common.cuh"

namespace gpu_ops {
namespace {

// 007 -----------------------------------------------------------------------
__global__ void op007_external_libdevice_asin_kernel(
    const float* x, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) out[index] = asinf(x[index]);
}

}  // namespace

cudaError_t launch_op007_external_libdevice_asin(
    const float* x, float* out, int n, cudaStream_t stream) {
  if (n < 0) return cudaErrorInvalidValue;
  if (n == 0) return cudaSuccess;
  if (x == nullptr || out == nullptr) return cudaErrorInvalidValue;
  op007_external_libdevice_asin_kernel<<<
      ceil_div_int(n, kThreads), kThreads, 0, stream>>>(x, out, n);
  return cudaGetLastError();
}

}  // namespace gpu_ops
