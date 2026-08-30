#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op080_greedy_decode_kernel(
    const float* logits, long long* out, int vocab) {
  extern __shared__ unsigned char storage[];
  float* best_values = reinterpret_cast<float*>(storage);
  int* best_indices = reinterpret_cast<int*>(best_values + blockDim.x);
  const int batch = blockIdx.x;
  float best = -FLT_MAX;
  int index = 0;
  for (int token = threadIdx.x; token < vocab; token += blockDim.x) {
    const float candidate = logits[batch * vocab + token];
    if (candidate > best || (candidate == best && token < index)) {
      best = candidate;
      index = token;
    }
  }
  best_values[threadIdx.x] = best;
  best_indices[threadIdx.x] = index;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other = best_values[threadIdx.x + stride];
      const int other_index = best_indices[threadIdx.x + stride];
      if (other > best_values[threadIdx.x] ||
          (other == best_values[threadIdx.x] &&
           other_index < best_indices[threadIdx.x])) {
        best_values[threadIdx.x] = other;
        best_indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) out[batch] = best_indices[0];
}

}  // namespace

cudaError_t launch_op080_greedy_decode(
    const float* logits, long long* out, int batch, int vocab,
    cudaStream_t stream) {
  if (batch < 0 || vocab <= 0) return cudaErrorInvalidValue;
  if (batch == 0) return cudaSuccess;
  const size_t shared = kThreads * (sizeof(float) + sizeof(int));
  op080_greedy_decode_kernel<<<batch, kThreads, shared, stream>>>(
      logits, out, vocab);
  return cudaGetLastError();
}

}  // namespace gpu_ops
