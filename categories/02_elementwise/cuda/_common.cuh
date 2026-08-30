#pragma once

#include "../../../shared/cuda_common.cuh"

#include <climits>

namespace gpu_ops {
namespace elementwise_detail {

inline cudaError_t validate_n(int n) {
  return n < 0 ? cudaErrorInvalidValue : cudaSuccess;
}

__device__ __forceinline__ float stable_sigmoid(float value) {
  const float exponential = expf(-fabsf(value));
  return value >= 0.0f ? 1.0f / (1.0f + exponential)
                       : exponential / (1.0f + exponential);
}

__device__ __forceinline__ float stable_softplus(float value) {
  return fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
}

}  // namespace elementwise_detail
}  // namespace gpu_ops
