#include "_common.cuh"

namespace gpu_ops {
namespace {

// A 2-D grid mirrors the Cartesian-product output coordinates.
__global__ void op059_outer_product_kernel(
    const float* x, const float* y, float* out, int rows, int cols) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols) out[row * cols + col] = x[row] * y[col];
}

}  // namespace

cudaError_t launch_op059_outer_product(
    const float* x, const float* y, float* out, int rows, int cols,
    cudaStream_t stream) {
  if (rows < 0 || cols < 0) return cudaErrorInvalidValue;
  if (rows == 0 || cols == 0) return cudaSuccess;
  dim3 block(linear_algebra_detail::kTile, linear_algebra_detail::kTile);
  dim3 grid(ceil_div_int(cols, linear_algebra_detail::kTile),
            ceil_div_int(rows, linear_algebra_detail::kTile));
  op059_outer_product_kernel<<<grid, block, 0, stream>>>(x, y, out, rows, cols);
  return cudaGetLastError();
}

}  // namespace gpu_ops
