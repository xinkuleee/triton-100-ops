#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op083_per_tensor_dequantize_kernel(
    const signed char* q, const float* scale, float* out, int count) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < count) out[offset] = static_cast<float>(q[offset]) * scale[0];
}
}  // namespace

cudaError_t launch_op083_per_tensor_dequantize(
    const signed char* q, const float* scale, float* out, int count,
    cudaStream_t stream) {
  if (count < 0) return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  op083_per_tensor_dequantize_kernel<<<
      ceil_div_int(count, kThreads), kThreads, 0, stream>>>(q, scale, out, count);
  return cudaGetLastError();
}
}  // namespace gpu_ops
