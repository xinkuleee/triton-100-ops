#include "../../../shared/cuda_common.cuh"

#include <cstdint>

namespace gpu_ops {
namespace {

// One CTA owns one requested embedding row. Threads walk contiguous D.
__global__ void op047_embedding_kernel(
    const float* weight, const int64_t* indices, float* out,
    int vocab, int width) {
  const int64_t index = indices[blockIdx.x];
  for (int col = threadIdx.x; col < width; col += blockDim.x)
    out[blockIdx.x * width + col] =
        index >= 0 && index < vocab ? weight[index * width + col] : 0.0f;
}

}  // namespace

cudaError_t launch_op047_embedding(
    const float* weight, const int64_t* indices, float* out,
    int count, int vocab, int width, cudaStream_t stream) {
  if (count < 0 || vocab < 0 || width <= 0) return cudaErrorInvalidValue;
  if (count == 0) return cudaSuccess;
  op047_embedding_kernel<<<count, kThreads, 0, stream>>>(
      weight, indices, out, vocab, width);
  return cudaGetLastError();
}

}  // namespace gpu_ops
