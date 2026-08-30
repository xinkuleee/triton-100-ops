#include "../../../shared/cuda_common.cuh"

#include <cuda_fp16.h>

#include <cfloat>
#include <climits>

namespace gpu_ops {
namespace {

// 006: one CTA owns a query. It streams K/V tiles and applies the online
// softmax recurrence, so no length-by-length score matrix is materialized.
constexpr int kAttentionTile = 32;
__global__ void op006_fused_attention_forward_kernel(
    const float* q, const float* k, const float* v, float* out, float* lse,
    int length, int dim, float scale, bool causal) {
  __shared__ float weights[kAttentionTile];
  __shared__ float running_max;
  __shared__ float running_sum;
  __shared__ float correction;
  __shared__ float tile_denominator;
  const int query = blockIdx.x;
  if (threadIdx.x == 0) {
    running_max = -FLT_MAX;
    running_sum = 0.0f;
  }
  for (int d = threadIdx.x; d < dim; d += blockDim.x)
    out[query * dim + d] = 0.0f;
  __syncthreads();

  for (int key0 = 0; key0 < length; key0 += kAttentionTile) {
    const int lane_key = threadIdx.x;
    const int key = key0 + lane_key;
    const bool valid = lane_key < kAttentionTile && key < length &&
                       (!causal || key <= query);
    float score = -FLT_MAX;
    if (valid) {
      score = 0.0f;
      for (int d = 0; d < dim; ++d)
        score += q[query * dim + d] * k[key * dim + d];
      score *= scale;
    }
    const float tile_max = block_max(score);
    if (threadIdx.x == 0) {
      const float next_max = fmaxf(running_max, tile_max);
      correction = expf(running_max - next_max);
      running_max = next_max;
    }
    __syncthreads();
    float weight = valid ? expf(score - running_max) : 0.0f;
    if (lane_key < kAttentionTile) weights[lane_key] = weight;
    const float tile_sum = block_sum(weight);
    if (threadIdx.x == 0) tile_denominator = tile_sum;
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      float tile_accumulator = 0.0f;
      #pragma unroll
      for (int inner = 0; inner < kAttentionTile; ++inner) {
        const int inner_key = key0 + inner;
        if (inner_key < length && (!causal || inner_key <= query))
          tile_accumulator += weights[inner] * v[inner_key * dim + d];
      }
      out[query * dim + d] =
          out[query * dim + d] * correction + tile_accumulator;
    }
    __syncthreads();
    if (threadIdx.x == 0)
      running_sum = running_sum * correction + tile_denominator;
    __syncthreads();
  }
  for (int d = threadIdx.x; d < dim; d += blockDim.x)
    out[query * dim + d] /= running_sum;
  if (threadIdx.x == 0) lse[query] = running_max + logf(running_sum);
}

// Teaching backward: query ownership is simple, while dK/dV conflicts are
// resolved with atomicAdd. Production kernels repartition tiles to avoid this.
__global__ void op006_fused_attention_backward_kernel(
    const float* q, const float* k, const float* v, const float* out,
    const float* dout, const float* lse, float* dq, float* dk, float* dv,
    int length, int dim, float scale, bool causal) {
  __shared__ float probability[kAttentionTile];
  __shared__ float ds[kAttentionTile];
  const int query = blockIdx.x;
  float local_delta = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x)
    local_delta += out[query * dim + d] * dout[query * dim + d];
  const float delta = block_sum(local_delta);
  const int owned_d = threadIdx.x;
  for (int d = owned_d; d < dim; d += blockDim.x)
    dq[query * dim + d] = 0.0f;
  __syncthreads();
  for (int key0 = 0; key0 < length; key0 += kAttentionTile) {
    const int lane_key = threadIdx.x;
    const int key = key0 + lane_key;
    const bool valid = lane_key < kAttentionTile && key < length &&
                       (!causal || key <= query);
    if (lane_key < kAttentionTile) {
      float score = 0.0f;
      float dp = 0.0f;
      if (valid) {
        for (int d = 0; d < dim; ++d) {
          score += q[query * dim + d] * k[key * dim + d];
          dp += v[key * dim + d] * dout[query * dim + d];
        }
      }
      const float p = valid ? expf(score * scale - lse[query]) : 0.0f;
      probability[lane_key] = p;
      ds[lane_key] = p * (dp - delta) * scale;
    }
    __syncthreads();
    for (int d = owned_d; d < dim; d += blockDim.x) {
      float tile_dq = 0.0f;
      #pragma unroll
      for (int inner = 0; inner < kAttentionTile; ++inner) {
        const int inner_key = key0 + inner;
        if (inner_key < length && (!causal || inner_key <= query)) {
          tile_dq += ds[inner] * k[inner_key * dim + d];
          atomicAdd(dk + inner_key * dim + d,
                    ds[inner] * q[query * dim + d]);
          atomicAdd(dv + inner_key * dim + d,
                    probability[inner] * dout[query * dim + d]);
        }
      }
      dq[query * dim + d] += tile_dq;
    }
    __syncthreads();
  }
}

// Batch/head-capable FP32 forward.  Flattening (batch, head, query) into
// blockIdx.x preserves the same online-softmax recurrence as the single-head
// teaching kernel without materializing the N_CTX x N_CTX score matrix.
__global__ void op006_batched_attention_forward_f32_kernel(
    const float* q, const float* k, const float* v, float* out, float* lse,
    int batches, int heads, int length, int dim, float scale, bool causal) {
  __shared__ float weights[kAttentionTile];
  __shared__ float running_max;
  __shared__ float running_sum;
  __shared__ float correction;
  __shared__ float tile_sum_shared;
  const int query = blockIdx.x % length;
  const int batch_head = blockIdx.x / length;
  if (batch_head >= batches * heads) return;
  const size_t base = static_cast<size_t>(batch_head) * length * dim;
  if (threadIdx.x == 0) {
    running_max = -FLT_MAX;
    running_sum = 0.0f;
  }
  for (int d = threadIdx.x; d < dim; d += blockDim.x)
    out[base + query * dim + d] = 0.0f;
  __syncthreads();
  for (int key0 = 0; key0 < length; key0 += kAttentionTile) {
    const int key = key0 + threadIdx.x;
    const bool valid = threadIdx.x < kAttentionTile && key < length &&
                       (!causal || key <= query);
    float score = -FLT_MAX;
    if (valid) {
      score = 0.0f;
      for (int d = 0; d < dim; ++d)
        score += q[base + query * dim + d] * k[base + key * dim + d];
      score *= scale;
    }
    const float tile_max = block_max(score);
    if (threadIdx.x == 0) {
      const float next_max = fmaxf(running_max, tile_max);
      correction = expf(running_max - next_max);
      running_max = next_max;
    }
    __syncthreads();
    const float weight = valid ? expf(score - running_max) : 0.0f;
    if (threadIdx.x < kAttentionTile) weights[threadIdx.x] = weight;
    const float tile_sum = block_sum(weight);
    if (threadIdx.x == 0) tile_sum_shared = tile_sum;
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      float tile_out = 0.0f;
      #pragma unroll
      for (int inner = 0; inner < kAttentionTile; ++inner) {
        const int inner_key = key0 + inner;
        if (inner_key < length && (!causal || inner_key <= query))
          tile_out += weights[inner] * v[base + inner_key * dim + d];
      }
      out[base + query * dim + d] =
          out[base + query * dim + d] * correction + tile_out;
    }
    __syncthreads();
    if (threadIdx.x == 0)
      running_sum = running_sum * correction + tile_sum_shared;
    __syncthreads();
  }
  for (int d = threadIdx.x; d < dim; d += blockDim.x)
    out[base + query * dim + d] /= running_sum;
  if (threadIdx.x == 0)
    lse[static_cast<size_t>(batch_head) * length + query] =
        running_max + logf(running_sum);
}

__global__ void op006_batched_attention_forward_f16_kernel(
    const half* q, const half* k, const half* v, half* out, float* lse,
    int batches, int heads, int length, int dim, float scale, bool causal) {
  extern __shared__ float accumulator[];
  __shared__ float weights[kAttentionTile];
  __shared__ float running_max;
  __shared__ float running_sum;
  __shared__ float correction;
  __shared__ float tile_sum_shared;
  const int query = blockIdx.x % length;
  const int batch_head = blockIdx.x / length;
  if (batch_head >= batches * heads) return;
  const size_t base = static_cast<size_t>(batch_head) * length * dim;
  if (threadIdx.x == 0) {
    running_max = -FLT_MAX;
    running_sum = 0.0f;
  }
  for (int d = threadIdx.x; d < dim; d += blockDim.x) accumulator[d] = 0.0f;
  __syncthreads();
  for (int key0 = 0; key0 < length; key0 += kAttentionTile) {
    const int key = key0 + threadIdx.x;
    const bool valid = threadIdx.x < kAttentionTile && key < length &&
                       (!causal || key <= query);
    float score = -FLT_MAX;
    if (valid) {
      score = 0.0f;
      for (int d = 0; d < dim; ++d)
        score += __half2float(q[base + query * dim + d]) *
                 __half2float(k[base + key * dim + d]);
      score *= scale;
    }
    const float tile_max = block_max(score);
    if (threadIdx.x == 0) {
      const float next_max = fmaxf(running_max, tile_max);
      correction = expf(running_max - next_max);
      running_max = next_max;
    }
    __syncthreads();
    const float weight = valid ? expf(score - running_max) : 0.0f;
    if (threadIdx.x < kAttentionTile) weights[threadIdx.x] = weight;
    const float tile_sum = block_sum(weight);
    if (threadIdx.x == 0) tile_sum_shared = tile_sum;
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      float tile_out = 0.0f;
      #pragma unroll
      for (int inner = 0; inner < kAttentionTile; ++inner) {
        const int inner_key = key0 + inner;
        if (inner_key < length && (!causal || inner_key <= query))
          tile_out += weights[inner] *
                      __half2float(v[base + inner_key * dim + d]);
      }
      accumulator[d] = accumulator[d] * correction + tile_out;
    }
    __syncthreads();
    if (threadIdx.x == 0)
      running_sum = running_sum * correction + tile_sum_shared;
    __syncthreads();
  }
  for (int d = threadIdx.x; d < dim; d += blockDim.x)
    out[base + query * dim + d] =
        __float2half(accumulator[d] / running_sum);
  if (threadIdx.x == 0)
    lse[static_cast<size_t>(batch_head) * length + query] =
        running_max + logf(running_sum);
}

// Atomics-free teaching backward for BxHxNxD FP32 tensors.  One kernel owns
// each query row and writes dQ; two independent kernels own each key row and
// write dK/dV.  This repeats score work but makes ownership deterministic and
// mirrors the official tutorial's split dK/dV and dQ stages.
__global__ void op006_attention_delta_f32_kernel(
    const float* out, const float* dout, float* delta, int total_rows,
    int dim) {
  const int row = blockIdx.x;
  if (row >= total_rows) return;
  float local = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x)
    local += out[row * dim + d] * dout[row * dim + d];
  const float sum = block_sum(local);
  if (threadIdx.x == 0) delta[row] = sum;
}

__global__ void op006_attention_dq_f32_kernel(
    const float* q, const float* k, const float* v, const float* dout,
    const float* lse, const float* delta, float* dq, int batches, int heads,
    int length, int dim, float scale, bool causal) {
  const int query = blockIdx.x;
  const int batch_head = query / length;
  const int query_in_head = query % length;
  if (batch_head >= batches * heads) return;
  const size_t base = static_cast<size_t>(batch_head) * length * dim;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float value = 0.0f;
    for (int key = 0; key < length; ++key) {
      if (causal && key > query_in_head) continue;
      float score = 0.0f;
      float dp = 0.0f;
      for (int inner = 0; inner < dim; ++inner) {
        score += q[base + query_in_head * dim + inner] *
                 k[base + key * dim + inner];
        dp += dout[base + query_in_head * dim + inner] *
              v[base + key * dim + inner];
      }
      const float probability = expf(
          score * scale - lse[batch_head * length + query_in_head]);
      const float ds = probability *
          (dp - delta[batch_head * length + query_in_head]) * scale;
      value += ds * k[base + key * dim + d];
    }
    dq[base + query_in_head * dim + d] = value;
  }
}

__global__ void op006_attention_dkdv_f32_kernel(
    const float* q, const float* k, const float* v, const float* dout,
    const float* lse, const float* delta, float* dk, float* dv, int batches,
    int heads, int length, int dim, float scale, bool causal) {
  const int key_row = blockIdx.x;
  const int batch_head = key_row / length;
  const int key = key_row % length;
  if (batch_head >= batches * heads) return;
  const size_t base = static_cast<size_t>(batch_head) * length * dim;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float dk_value = 0.0f;
    float dv_value = 0.0f;
    for (int query = 0; query < length; ++query) {
      if (causal && key > query) continue;
      float score = 0.0f;
      float dp = 0.0f;
      for (int inner = 0; inner < dim; ++inner) {
        score += q[base + query * dim + inner] *
                 k[base + key * dim + inner];
        dp += dout[base + query * dim + inner] *
              v[base + key * dim + inner];
      }
      const float probability =
          expf(score * scale - lse[batch_head * length + query]);
      const float ds = probability *
          (dp - delta[batch_head * length + query]) * scale;
      dk_value += ds * q[base + query * dim + d];
      dv_value += probability * dout[base + query * dim + d];
    }
    dk[base + key * dim + d] = dk_value;
    dv[base + key * dim + d] = dv_value;
  }
}

}  // namespace

cudaError_t launch_op006_fused_attention_forward(
    const float* q, const float* k, const float* v, float* out, float* lse,
    int length, int dim, bool causal, cudaStream_t stream) {
  if (length <= 0 || dim <= 0) return cudaErrorInvalidValue;
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
      lse == nullptr)
    return cudaErrorInvalidValue;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
  op006_fused_attention_forward_kernel<<<length, 128, 0, stream>>>(
      q, k, v, out, lse, length, dim, scale, causal);
  return cudaGetLastError();
}

cudaError_t launch_op006_fused_attention_backward(
    const float* q, const float* k, const float* v, const float* out,
    const float* dout, const float* lse, float* dq, float* dk, float* dv,
    int length, int dim, bool causal, cudaStream_t stream) {
  if (length <= 0 || dim <= 0) return cudaErrorInvalidValue;
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
      dout == nullptr || lse == nullptr || dq == nullptr || dk == nullptr ||
      dv == nullptr)
    return cudaErrorInvalidValue;
  const size_t gradient_bytes =
      static_cast<size_t>(length) * static_cast<size_t>(dim) * sizeof(float);
  CUDA_CHECK(cudaMemsetAsync(dk, 0, gradient_bytes, stream));
  CUDA_CHECK(cudaMemsetAsync(dv, 0, gradient_bytes, stream));
  const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
  op006_fused_attention_backward_kernel<<<length, 128, 0, stream>>>(
      q, k, v, out, dout, lse, dq, dk, dv, length, dim, scale, causal);
  return cudaGetLastError();
}

cudaError_t launch_op006_batched_attention_forward_f32(
    const float* q, const float* k, const float* v, float* out, float* lse,
    int batches, int heads, int length, int dim, bool causal,
    cudaStream_t stream) {
  if (batches < 0 || heads <= 0 || length <= 0 || dim <= 0)
    return cudaErrorInvalidValue;
  if (batches == 0) return cudaSuccess;
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
      lse == nullptr)
    return cudaErrorInvalidValue;
  const int64_t blocks =
      static_cast<int64_t>(batches) * heads * length;
  if (blocks > INT_MAX) return cudaErrorInvalidValue;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
  op006_batched_attention_forward_f32_kernel<<<
      static_cast<int>(blocks), 128, 0, stream>>>(
      q, k, v, out, lse, batches, heads, length, dim, scale, causal);
  return cudaGetLastError();
}

cudaError_t launch_op006_batched_attention_forward_f16(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out,
    float* lse,
    int batches, int heads, int length, int dim, bool causal,
    cudaStream_t stream) {
  if (batches < 0 || heads <= 0 || length <= 0 || dim <= 0)
    return cudaErrorInvalidValue;
  if (batches == 0) return cudaSuccess;
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
      lse == nullptr)
    return cudaErrorInvalidValue;
  const int64_t blocks =
      static_cast<int64_t>(batches) * heads * length;
  if (blocks > INT_MAX) return cudaErrorInvalidValue;
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  int max_shared = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shared, cudaDevAttrMaxSharedMemoryPerBlock, device));
  if (static_cast<size_t>(dim) * sizeof(float) >
      static_cast<size_t>(max_shared))
    return cudaErrorNotSupported;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
  op006_batched_attention_forward_f16_kernel<<<
      static_cast<int>(blocks), 128, dim * sizeof(float), stream>>>(
      reinterpret_cast<const half*>(q), reinterpret_cast<const half*>(k),
      reinterpret_cast<const half*>(v), reinterpret_cast<half*>(out), lse,
      batches, heads, length, dim, scale, causal);
  return cudaGetLastError();
}

cudaError_t launch_op006_batched_attention_backward_f32_staged(
    const float* q, const float* k, const float* v, const float* out,
    const float* dout, const float* lse, float* delta, float* dq, float* dk,
    float* dv, int batches, int heads, int length, int dim, bool causal,
    cudaStream_t stream) {
  if (batches < 0 || heads <= 0 || length <= 0 || dim <= 0)
    return cudaErrorInvalidValue;
  if (batches == 0) return cudaSuccess;
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
      dout == nullptr || lse == nullptr || delta == nullptr || dq == nullptr ||
      dk == nullptr || dv == nullptr)
    return cudaErrorInvalidValue;
  const int64_t total_rows_64 =
      static_cast<int64_t>(batches) * heads * length;
  if (total_rows_64 > INT_MAX) return cudaErrorInvalidValue;
  const int total_rows = static_cast<int>(total_rows_64);
  const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
  op006_attention_delta_f32_kernel<<<total_rows, 128, 0, stream>>>(
      out, dout, delta, total_rows, dim);
  CUDA_CHECK(cudaGetLastError());
  op006_attention_dq_f32_kernel<<<total_rows, 128, 0, stream>>>(
      q, k, v, dout, lse, delta, dq, batches, heads, length, dim, scale,
      causal);
  CUDA_CHECK(cudaGetLastError());
  op006_attention_dkdv_f32_kernel<<<total_rows, 128, 0, stream>>>(
      q, k, v, dout, lse, delta, dk, dv, batches, heads, length, dim, scale,
      causal);
  return cudaGetLastError();
}

}  // namespace gpu_ops
