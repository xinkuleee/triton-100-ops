#include "_common.cuh"

namespace gpu_ops {
namespace {

// Adjacent even/odd features form one rotation pair.
__global__ void op070_rope_kernel(
    const float* x, const float* cosine, const float* sine, float* out,
    int pair_count, int heads, int dim) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= pair_count) return;
  const int half = dim / 2;
  const int frequency = pair % half;
  const int head = (pair / half) % heads;
  const int token = pair / (half * heads);
  const int base = (token * heads + head) * dim + 2 * frequency;
  const float even = x[base];
  const float odd = x[base + 1];
  const float c = cosine[token * half + frequency];
  const float s = sine[token * half + frequency];
  out[base] = even * c - odd * s;
  out[base + 1] = even * s + odd * c;
}

}  // namespace

cudaError_t launch_op070_rope(
    const float* x, const float* cosine, const float* sine, float* out,
    int tokens, int heads, int dim, cudaStream_t stream) {
  if (tokens < 0 || heads < 0 || dim <= 0 || dim % 2)
    return cudaErrorInvalidValue;
  const long long wide = 1LL * tokens * heads * (dim / 2);
  if (wide == 0) return cudaSuccess;
  if (wide > INT_MAX) return cudaErrorInvalidValue;
  const int pairs = static_cast<int>(wide);
  op070_rope_kernel<<<ceil_div_int(pairs, kThreads), kThreads, 0, stream>>>(
      x, cosine, sine, out, pairs, heads, dim);
  return cudaGetLastError();
}

}  // namespace gpu_ops

