#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op090_bce_with_logits_kernel(
    const float* logits, const float* targets, float* out, int count) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= count) return;
  const float value = logits[offset];
  out[offset] = fmaxf(value, 0.0f) - value * targets[offset] +
                log1pf(expf(-fabsf(value)));
}

}  // namespace

cudaError_t launch_op090_bce_with_logits(
    const float* logits, const float* targets, float* out, int count,
    cudaStream_t stream) {
  if (count < 0) return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  op090_bce_with_logits_kernel<<<
      ceil_div_int(count, kThreads), kThreads, 0, stream>>>(
      logits, targets, out, count);
  return cudaGetLastError();
}

}  // namespace gpu_ops
