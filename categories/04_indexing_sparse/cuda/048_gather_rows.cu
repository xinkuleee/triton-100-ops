#include "../../../shared/cuda_common.cuh"

#include <climits>
#include <cstdint>

namespace gpu_ops {
namespace {

// Output threads gather arbitrary columns within their logical row.
__global__ void op048_gather_rows_kernel(
    const float* x, const int64_t* indices, float* out,
    int total, int cols, int count) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= total) return;
  const int row = offset / count;
  const int64_t index = indices[offset];
  out[offset] = index >= 0 && index < cols ? x[row * cols + index] : 0.0f;
}

inline bool op048_product_fits_int(int left, int right) {
  return left == 0 || right <= INT_MAX / left;
}

}  // namespace

cudaError_t launch_op048_gather_rows(
    const float* x, const int64_t* indices, float* out,
    int rows, int cols, int count, cudaStream_t stream) {
  if (rows < 0 || cols <= 0 || count < 0 ||
      !op048_product_fits_int(rows, count)) return cudaErrorInvalidValue;
  const int total = rows * count;
  if (total == 0) return cudaSuccess;
  op048_gather_rows_kernel<<<ceil_div_int(total, kThreads), kThreads, 0, stream>>>(
      x, indices, out, total, cols, count);
  return cudaGetLastError();
}

}  // namespace gpu_ops
