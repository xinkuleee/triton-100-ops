#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op079_temperature_scale_kernel(
    const float* logits, float* out, int count, float temperature) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < count) out[offset] = logits[offset] / temperature;
}

}  // namespace

cudaError_t launch_op079_temperature_scale(
    const float* logits, float* out, int count, float temperature,
    cudaStream_t stream) {
  if (count < 0 || temperature <= 0.0f || !isfinite(temperature))
    return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  op079_temperature_scale_kernel<<<
      ceil_div_int(count, kThreads), kThreads, 0, stream>>>(
      logits, out, count, temperature);
  return cudaGetLastError();
}

}  // namespace gpu_ops
