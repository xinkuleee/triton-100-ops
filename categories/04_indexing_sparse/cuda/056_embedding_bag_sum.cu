#include "../../../shared/cuda_common.cuh"

namespace gpu_ops {
namespace {

// One CTA owns a bag; threads independently accumulate embedding columns.
__global__ void op056_embedding_bag_sum_kernel(
    const float* weight, const int* indices, const int* offsets, float* out,
    int bags, int vocab, int width) {
  const int bag = blockIdx.x;
  if (bag >= bags) return;
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    float sum = 0.0f;
    for (int position = offsets[bag]; position < offsets[bag + 1]; ++position) {
      const int index = indices[position];
      if (index >= 0 && index < vocab)
        sum += weight[index * width + col];
    }
    out[bag * width + col] = sum;
  }
}

}  // namespace

cudaError_t launch_op056_embedding_bag_sum(
    const float* weight, const int* indices, const int* offsets, float* out,
    int bags, int vocab, int width, cudaStream_t stream) {
  if (bags < 0 || vocab < 0 || width <= 0) return cudaErrorInvalidValue;
  if (bags == 0) return cudaSuccess;
  op056_embedding_bag_sum_kernel<<<bags, kThreads, 0, stream>>>(
      weight, indices, offsets, out, bags, vocab, width);
  return cudaGetLastError();
}

}  // namespace gpu_ops
