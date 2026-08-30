#include "_common.cuh"

namespace gpu_ops {
namespace {

// positions[B,T] maps incoming tokens to logical cache slots.
__global__ void op072_kv_cache_append_kernel(
    const float* new_key, const float* new_value, const long long* positions,
    float* cache_key, float* cache_value, int total, int heads, int tokens,
    int dim, int max_sequence) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= total) return;
  const int d = offset % dim;
  const int token = (offset / dim) % tokens;
  const int head = (offset / (dim * tokens)) % heads;
  const int batch = offset / (dim * tokens * heads);
  const long long position = positions[batch * tokens + token];
  if (position < 0 || position >= max_sequence) return;
  const long long destination =
      ((static_cast<long long>(batch) * heads + head) * max_sequence +
       position) * dim + d;
  cache_key[destination] = new_key[offset];
  cache_value[destination] = new_value[offset];
}

}  // namespace

cudaError_t launch_op072_kv_cache_append(
    const float* new_key, const float* new_value, const long long* positions,
    float* cache_key, float* cache_value, int batch, int heads, int tokens,
    int dim, int max_sequence, cudaStream_t stream) {
  if (batch < 0 || heads < 0 || tokens < 0 || dim < 0 || max_sequence <= 0)
    return cudaErrorInvalidValue;
  const long long wide = 1LL * batch * heads * tokens * dim;
  if (wide == 0) return cudaSuccess;
  if (wide > INT_MAX) return cudaErrorInvalidValue;
  const int total = static_cast<int>(wide);
  op072_kv_cache_append_kernel<<<
      ceil_div_int(total, kThreads), kThreads, 0, stream>>>(
      new_key, new_value, positions, cache_key, cache_value, total, heads,
      tokens, dim, max_sequence);
  return cudaGetLastError();
}

}  // namespace gpu_ops

