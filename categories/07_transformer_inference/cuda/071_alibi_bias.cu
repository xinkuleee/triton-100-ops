#include "_common.cuh"

namespace gpu_ops {
namespace {

// scores[b,h,q,k] += slopes[h] * (k-q).
__global__ void op071_alibi_bias_kernel(
    const float* scores, const float* slopes, float* out, int total,
    int heads, int queries, int keys) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= total) return;
  const int key = offset % keys;
  const int query = (offset / keys) % queries;
  const int head = (offset / (keys * queries)) % heads;
  out[offset] = scores[offset] +
                slopes[head] * static_cast<float>(key - query);
}

}  // namespace

cudaError_t launch_op071_alibi_bias(
    const float* scores, const float* slopes, float* out, int batch, int heads,
    int queries, int keys, cudaStream_t stream) {
  if (batch < 0 || heads < 0 || queries < 0 || keys < 0)
    return cudaErrorInvalidValue;
  const long long wide = 1LL * batch * heads * queries * keys;
  if (wide == 0) return cudaSuccess;
  if (wide > INT_MAX) return cudaErrorInvalidValue;
  const int total = static_cast<int>(wide);
  op071_alibi_bias_kernel<<<ceil_div_int(total, kThreads), kThreads, 0, stream>>>(
      scores, slopes, out, total, heads, queries, keys);
  return cudaGetLastError();
}

}  // namespace gpu_ops

