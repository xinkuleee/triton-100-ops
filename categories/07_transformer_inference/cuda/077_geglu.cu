#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op077_geglu_kernel(
    const float* gate, const float* value, float* out, int count) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= count) return;
  const float x = gate[offset];
  const float inner = 0.7978845608028654f *
                      (x + 0.044715f * x * x * x);
  const float gelu = 0.5f * x * (1.0f + tanhf(inner));
  out[offset] = gelu * value[offset];
}

}  // namespace

cudaError_t launch_op077_geglu(
    const float* gate, const float* value, float* out, int count,
    cudaStream_t stream) {
  if (count < 0) return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  op077_geglu_kernel<<<ceil_div_int(count, kThreads), kThreads, 0, stream>>>(
      gate, value, out, count);
  return cudaGetLastError();
}

}  // namespace gpu_ops
