#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op082_per_tensor_quantize_kernel(
    const float* x, const float* scale, signed char* q, int count) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < count)
    q[offset] = quantization_detail::op082_quantize_value(x[offset], scale[0]);
}
}  // namespace

cudaError_t launch_op082_per_tensor_quantize(
    const float* x, const float* scale, signed char* q, int count,
    cudaStream_t stream) {
  if (count < 0) return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  op082_per_tensor_quantize_kernel<<<
      ceil_div_int(count, kThreads), kThreads, 0, stream>>>(x, scale, q, count);
  return cudaGetLastError();
}
}  // namespace gpu_ops
