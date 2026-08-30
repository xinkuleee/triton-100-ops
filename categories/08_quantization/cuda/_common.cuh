#pragma once

#include "../../../shared/cuda_common.cuh"

#include <climits>

namespace gpu_ops {
namespace quantization_detail {

constexpr int kMaxSafeInt8DotK = 131071;

__device__ __forceinline__ float op082_symmetric_scale(float absolute_maximum) {
  return fmaxf(absolute_maximum / 127.0f, 1.0e-12f);
}

__device__ __forceinline__ signed char op082_quantize_value(
    float value, float scale) {
  const float clipped = fminf(127.0f, fmaxf(-127.0f, value / scale));
  return static_cast<signed char>(roundf(clipped));
}

}  // namespace quantization_detail
}  // namespace gpu_ops
