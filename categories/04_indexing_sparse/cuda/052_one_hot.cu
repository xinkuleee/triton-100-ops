#include "../../../shared/cuda_common.cuh"

#include <climits>
#include <cstdint>

namespace gpu_ops {
namespace {

// Each thread writes one class slot, so no initialization kernel is needed.
__global__ void op052_one_hot_kernel(
    const int64_t* indices, float* out, int total, int classes) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= total) return;
  const int item = offset / classes;
  const int cls = offset % classes;
  out[offset] = indices[item] == cls ? 1.0f : 0.0f;
}

inline bool op052_product_fits_int(int left, int right) {
  return left == 0 || right <= INT_MAX / left;
}

}  // namespace

cudaError_t launch_op052_one_hot(
    const int64_t* indices, float* out, int count, int classes,
    cudaStream_t stream) {
  if (count < 0 || classes <= 0 ||
      !op052_product_fits_int(count, classes)) return cudaErrorInvalidValue;
  const int total = count * classes;
  if (total == 0) return cudaSuccess;
  op052_one_hot_kernel<<<ceil_div_int(total, kThreads), kThreads, 0, stream>>>(
      indices, out, total, classes);
  return cudaGetLastError();
}

}  // namespace gpu_ops
