#include "_common.cuh"

namespace gpu_ops {
namespace {

inline int op069_next_power_of_two_threads(int value) {
  int threads = 32;
  while (threads < value && threads < 1024) threads <<= 1;
  return threads;
}

// One CTA owns one causal attention row.
__global__ void op069_causal_attention_kernel(
    const float* q, const float* key, const float* value, float* out,
    int query_length, int key_length, int dim, float scale) {
  extern __shared__ float probabilities[];
  const int row = blockIdx.x;
  const int query_index = row % query_length;
  const int batch_head = row / query_length;
  const int lane = threadIdx.x;

  float score = -INFINITY;
  if (lane < key_length && lane <= query_index) {
    score = 0.0f;
    for (int d = 0; d < dim; ++d)
      score += q[(batch_head * query_length + query_index) * dim + d] *
               key[(batch_head * key_length + lane) * dim + d];
    score *= scale;
  }
  const float maximum = block_max(score);
  const float exponential = lane < key_length && lane <= query_index
                                ? expf(score - maximum)
                                : 0.0f;
  const float denominator = block_sum(exponential);
  if (lane < key_length) probabilities[lane] = exponential;
  __syncthreads();

  if (lane < dim) {
    float numerator = 0.0f;
    for (int token = 0; token < key_length; ++token)
      numerator += probabilities[token] *
                   value[(batch_head * key_length + token) * dim + lane];
    out[(batch_head * query_length + query_index) * dim + lane] =
        numerator / denominator;
  }
}

}  // namespace

cudaError_t launch_op069_causal_attention(
    const float* q, const float* key, const float* value, float* out,
    int batch, int heads, int query_length, int key_length, int dim,
    cudaStream_t stream) {
  if (batch < 0 || heads < 0 || query_length < 0 || key_length <= 0 ||
      dim <= 0 || key_length > 1024 || dim > 1024)
    return cudaErrorInvalidValue;
  if (batch == 0 || heads == 0 || query_length == 0) return cudaSuccess;
  const int threads =
      op069_next_power_of_two_threads(max(key_length, dim));
  op069_causal_attention_kernel<<<
      batch * heads * query_length, threads, key_length * sizeof(float), stream>>>(
      q, key, value, out, query_length, key_length, dim,
      1.0f / sqrtf(static_cast<float>(dim)));
  return cudaGetLastError();
}

}  // namespace gpu_ops

