#include "_common.cuh"

namespace gpu_ops {
namespace {

inline int op073_next_power_of_two_threads(int value) {
  int threads = 32;
  while (threads < value && threads < 1024) threads <<= 1;
  return threads;
}

// page_table maps each logical token to a physical cache page.
__global__ void op073_paged_attention_kernel(
    const float* q, const float* key_cache, const float* value_cache,
    const int* page_table, const int* lengths, float* out, int heads,
    int dim, int page_size, int max_pages, int max_sequence, float scale) {
  extern __shared__ float probabilities[];
  const int batch_head = blockIdx.x;
  const int batch = batch_head / heads;
  const int head = batch_head % heads;
  const int lane = threadIdx.x;
  const int length = max(0, min(lengths[batch], max_sequence));
  if (length == 0) {
    if (lane < dim) out[batch_head * dim + lane] = 0.0f;
    return;
  }

  float score = -INFINITY;
  if (lane < length) {
    const int page = page_table[batch * max_pages + lane / page_size];
    const long long base =
        ((static_cast<long long>(page) * page_size + lane % page_size) *
         heads + head) * dim;
    score = 0.0f;
    for (int d = 0; d < dim; ++d)
      score += q[batch_head * dim + d] * key_cache[base + d];
    score *= scale;
  }
  const float maximum = block_max(score);
  const float exponential = lane < length ? expf(score - maximum) : 0.0f;
  const float denominator = block_sum(exponential);
  if (lane < length) probabilities[lane] = exponential;
  __syncthreads();

  if (lane < dim) {
    float numerator = 0.0f;
    for (int token = 0; token < length; ++token) {
      const int page =
          page_table[batch * max_pages + token / page_size];
      const long long base =
          ((static_cast<long long>(page) * page_size + token % page_size) *
           heads + head) * dim;
      numerator += probabilities[token] * value_cache[base + lane];
    }
    out[batch_head * dim + lane] = numerator / denominator;
  }
}

}  // namespace

cudaError_t launch_op073_paged_attention(
    const float* q, const float* key_cache, const float* value_cache,
    const int* page_table, const int* lengths, float* out, int batch, int heads,
    int dim, int page_size, int max_pages, int max_sequence,
    cudaStream_t stream) {
  if (batch < 0 || heads < 0 || dim <= 0 || page_size <= 0 ||
      max_pages <= 0 || max_sequence <= 0 || dim > 1024 ||
      max_sequence > 1024 || max_sequence > page_size * max_pages)
    return cudaErrorInvalidValue;
  if (batch == 0 || heads == 0) return cudaSuccess;
  const int threads =
      op073_next_power_of_two_threads(max(dim, max_sequence));
  op073_paged_attention_kernel<<<
      batch * heads, threads, max_sequence * sizeof(float), stream>>>(
      q, key_cache, value_cache, page_table, lengths, out, heads, dim,
      page_size, max_pages, max_sequence,
      1.0f / sqrtf(static_cast<float>(dim)));
  return cudaGetLastError();
}

}  // namespace gpu_ops

