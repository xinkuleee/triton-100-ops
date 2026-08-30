#include "../../../shared/cuda_common.cuh"

#include <climits>
#include <cstdint>

namespace gpu_ops {
namespace {

// Duplicate destinations race by definition; use 050 when addition is wanted.
__global__ void op049_scatter_rows_kernel(
    const float* src, const int64_t* indices, float* out,
    int total, int cols, int count) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= total) return;
  const int row = offset / count;
  const int64_t index = indices[offset];
  if (index >= 0 && index < cols) out[row * cols + index] = src[offset];
}

inline bool op049_product_fits_int(int left, int right) {
  return left == 0 || right <= INT_MAX / left;
}

}  // namespace

cudaError_t launch_op049_scatter_rows(
    const float* src, const int64_t* indices, float* out,
    int rows, int cols, int count, cudaStream_t stream) {
  if (rows < 0 || cols <= 0 || count < 0 ||
      !op049_product_fits_int(rows, count)) return cudaErrorInvalidValue;
  const int total = rows * count;
  if (total == 0) return cudaSuccess;
  op049_scatter_rows_kernel<<<ceil_div_int(total, kThreads), kThreads, 0, stream>>>(
      src, indices, out, total, cols, count);
  return cudaGetLastError();
}

}  // namespace gpu_ops
