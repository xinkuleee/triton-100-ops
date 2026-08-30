#pragma once

#include "../../../shared/cuda_common.cuh"

#include <climits>

namespace gpu_ops {
namespace vision_detail {

inline bool spatial_valid(
    int height, int width, int kernel_height, int kernel_width,
    int stride, int padding) {
  return height > 0 && width > 0 && kernel_height > 0 && kernel_width > 0 &&
         stride > 0 && padding >= 0;
}

inline bool total_fits(long long total) {
  return total >= 0 && total <= INT_MAX;
}

}  // namespace vision_detail
}  // namespace gpu_ops
