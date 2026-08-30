#pragma once

#include "../../../shared/cuda_common.cuh"

#include <climits>
#include <cmath>
#include <cstdint>
#include <math_constants.h>

namespace gpu_ops {
namespace reductions_detail {

inline cudaError_t validate_matrix(int rows, int cols) {
  if (rows < 0 || cols <= 0 || rows > INT_MAX / cols)
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

}  // namespace reductions_detail
}  // namespace gpu_ops
