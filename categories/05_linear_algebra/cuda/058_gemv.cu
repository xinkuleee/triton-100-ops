#include "_common.cuh"

namespace gpu_ops {
namespace {

// One CTA owns a row; block_sum uses warp shuffle then shared memory.
__global__ void op058_gemv_kernel(
    const float* matrix, const float* vector, float* out, int rows, int cols) {
  const int row = blockIdx.x;
  float partial = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    partial += matrix[row * cols + col] * vector[col];
  const float sum = block_sum(partial);
  if (threadIdx.x == 0 && row < rows) out[row] = sum;
}

}  // namespace

cudaError_t launch_op058_gemv(
    const float* matrix, const float* vector, float* out, int rows, int cols,
    cudaStream_t stream) {
  if (rows < 0 || cols <= 0 || cols > linear_algebra_detail::kMaxDotK)
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  op058_gemv_kernel<<<rows, kThreads, 0, stream>>>(matrix, vector, out, rows, cols);
  return cudaGetLastError();
}

}  // namespace gpu_ops
