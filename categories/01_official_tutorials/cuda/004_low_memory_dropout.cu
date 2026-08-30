#include "../../../shared/cuda_common.cuh"

#include <curand_kernel.h>

namespace gpu_ops {
namespace {

// 004: cuRAND's Philox state is indexed by the global element counter. This is
// the same stateless-recomputation contract as tl.rand, not a bitwise promise.
__global__ void op004_mask_dropout_kernel(
    const float* x, const bool* keep_mask, float* out, int n,
    float probability) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n)
    out[index] = keep_mask[index] ? x[index] / (1.0f - probability) : 0.0f;
}

__global__ void op004_low_memory_dropout_kernel(
    const float* x, float* out, int n, float probability, uint64_t seed) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, 0, static_cast<uint64_t>(index), &state);
  const bool keep = curand_uniform(&state) > probability;
  out[index] = keep ? x[index] / (1.0f - probability) : 0.0f;
}

}  // namespace

cudaError_t launch_op004_low_memory_dropout(
    const float* x, float* out, int n, float probability, uint64_t seed,
    cudaStream_t stream) {
  if (n < 0 || !std::isfinite(probability) || probability < 0.0f ||
      probability >= 1.0f)
    return cudaErrorInvalidValue;
  if (n == 0) return cudaSuccess;
  if (x == nullptr || out == nullptr) return cudaErrorInvalidValue;
  op004_low_memory_dropout_kernel<<<
      ceil_div_int(n, kThreads), kThreads, 0, stream>>>(
      x, out, n, probability, seed);
  return cudaGetLastError();
}

// Public launcher for the official tutorial's explicit-mask baseline.
cudaError_t launch_op004_explicit_mask_dropout(
    const float* x, const bool* keep_mask, float* out, int n,
    float probability, cudaStream_t stream) {
  if (n < 0 || !std::isfinite(probability) || probability < 0.0f ||
      probability >= 1.0f)
    return cudaErrorInvalidValue;
  if (n == 0) return cudaSuccess;
  if (x == nullptr || keep_mask == nullptr || out == nullptr)
    return cudaErrorInvalidValue;
  op004_mask_dropout_kernel<<<ceil_div_int(n, kThreads), kThreads, 0, stream>>>(
      x, keep_mask, out, n, probability);
  return cudaGetLastError();
}

}  // namespace gpu_ops
