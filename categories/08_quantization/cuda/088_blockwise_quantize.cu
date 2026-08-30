#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op088_blockwise_quantize_kernel(
    const float* x, signed char* q, float* scales, int count, int block_size) {
  const int start = blockIdx.x * block_size;
  float local_maximum = 0.0f;
  for (int offset = threadIdx.x; offset < block_size && start + offset < count;
       offset += blockDim.x)
    local_maximum = fmaxf(local_maximum, fabsf(x[start + offset]));
  const float scale =
      quantization_detail::op082_symmetric_scale(block_max(local_maximum));
  for (int offset = threadIdx.x; offset < block_size && start + offset < count;
       offset += blockDim.x)
    q[start + offset] =
        quantization_detail::op082_quantize_value(x[start + offset], scale);
  if (threadIdx.x == 0) scales[blockIdx.x] = scale;
}
}  // namespace

cudaError_t launch_op088_blockwise_quantize(
    const float* x, signed char* q, float* scales, int count, int block_size,
    cudaStream_t stream) {
  if (count < 0 || block_size <= 0 || block_size > 65536 ||
      (block_size & (block_size - 1)) != 0)
    return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  const int blocks = ceil_div_int(count, block_size);
  op088_blockwise_quantize_kernel<<<blocks, kThreads, 0, stream>>>(
      x, q, scales, count, block_size);
  return cudaGetLastError();
}
}  // namespace gpu_ops
