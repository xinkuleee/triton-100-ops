#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op087_fake_quantize_kernel(
    const float* x, const float* scale, float* out, int count) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < count)
    out[offset] = static_cast<float>(
                      quantization_detail::op082_quantize_value(x[offset], scale[0])) *
                  scale[0];
}
}  // namespace

cudaError_t launch_op087_fake_quantize(
    const float* x, const float* scale, float* out, int count,
    cudaStream_t stream) {
  if (count < 0) return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  op087_fake_quantize_kernel<<<
      ceil_div_int(count, kThreads), kThreads, 0, stream>>>(
      x, scale, out, count);
  return cudaGetLastError();
}
}  // namespace gpu_ops
