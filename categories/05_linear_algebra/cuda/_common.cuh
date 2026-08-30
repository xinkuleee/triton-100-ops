#pragma once

#include "../../../shared/cuda_common.cuh"

#include <climits>

namespace gpu_ops {
namespace linear_algebra_detail {

constexpr int kTile = 16;
constexpr int kMaxDotK = 65536;

inline bool matrix_dims_valid(int m, int n, int k) {
  if (m < 0 || n < 0 || k < 0) return false;
  if (m != 0 && n > INT_MAX / m) return false;
  return true;
}

}  // namespace linear_algebra_detail
}  // namespace gpu_ops
