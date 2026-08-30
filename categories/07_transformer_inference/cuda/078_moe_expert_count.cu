#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op078_moe_expert_count_kernel(
    const int* expert_ids, int* counts, int tokens, int experts) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < tokens) {
    const int expert = expert_ids[offset];
    if (expert >= 0 && expert < experts) atomicAdd(counts + expert, 1);
  }
}

}  // namespace

cudaError_t launch_op078_moe_expert_count(
    const int* expert_ids, int* counts, int tokens, int experts,
    cudaStream_t stream) {
  if (tokens < 0 || experts <= 0) return cudaErrorInvalidValue;
  CUDA_CHECK(cudaMemsetAsync(counts, 0,
                             static_cast<size_t>(experts) * sizeof(int), stream));
  if (tokens == 0) return cudaSuccess;
  op078_moe_expert_count_kernel<<<
      ceil_div_int(tokens, kThreads), kThreads, 0, stream>>>(
      expert_ids, counts, tokens, experts);
  return cudaGetLastError();
}

}  // namespace gpu_ops
