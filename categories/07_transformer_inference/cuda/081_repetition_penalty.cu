#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op081_repetition_penalty_kernel(
    float* logits, const int* token_ids, int history, int vocab,
    float penalty) {
  const int batch = blockIdx.y;
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= history) return;
  const int token = token_ids[batch * history + offset];
  if (token < 0 || token >= vocab) return;
  // Contract: token_ids are unique within each batch row.
  const float value = logits[batch * vocab + token];
  logits[batch * vocab + token] =
      value > 0.0f ? value / penalty : value * penalty;
}

}  // namespace

cudaError_t launch_op081_repetition_penalty(
    float* logits, const int* token_ids, int batch, int history, int vocab,
    float penalty, cudaStream_t stream) {
  if (batch < 0 || history < 0 || vocab <= 0 || penalty <= 0.0f ||
      !isfinite(penalty))
    return cudaErrorInvalidValue;
  if (batch == 0 || history == 0) return cudaSuccess;
  const dim3 grid(ceil_div_int(history, kThreads), batch);
  op081_repetition_penalty_kernel<<<grid, kThreads, 0, stream>>>(
      logits, token_ids, history, vocab, penalty);
  return cudaGetLastError();
}

}  // namespace gpu_ops
