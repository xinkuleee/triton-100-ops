// 035 Row Mean: one CTA sums one FP32 row and divides by its width.
#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op035_row_mean_kernel(const float* x, float* out, int cols) {
  float local = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local += x[blockIdx.x * cols + col];
  const float result = block_sum(local);
  if (threadIdx.x == 0) out[blockIdx.x] = result / cols;
}
}  // namespace

cudaError_t launch_op035_row_mean(const float* x, float* out, int rows, int cols, cudaStream_t stream) {
  CUDA_CHECK(reductions_detail::validate_matrix(rows, cols));
  if (rows == 0) return cudaSuccess;
  op035_row_mean_kernel<<<rows, kThreads, 0, stream>>>(x, out, cols);
  return cudaGetLastError();
}
}  // namespace gpu_ops
