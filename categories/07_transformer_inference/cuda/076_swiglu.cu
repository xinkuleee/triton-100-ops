#include "_common.cuh"

namespace gpu_ops {
namespace {

__device__ __forceinline__ float op076_stable_sigmoid(float value) {
  const float exponential = expf(-fabsf(value));
  return value >= 0.0f ? 1.0f / (1.0f + exponential)
                       : exponential / (1.0f + exponential);
}

__global__ void op076_swiglu_kernel(
    const float* gate, const float* value, float* out, int count) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < count)
    out[offset] = gate[offset] * op076_stable_sigmoid(gate[offset]) *
                  value[offset];
}

}  // namespace

cudaError_t launch_op076_swiglu(
    const float* gate, const float* value, float* out, int count,
    cudaStream_t stream) {
  if (count < 0) return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  op076_swiglu_kernel<<<ceil_div_int(count, kThreads), kThreads, 0, stream>>>(
      gate, value, out, count);
  return cudaGetLastError();
}

}  // namespace gpu_ops
