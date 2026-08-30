// 040 Variance: a two-pass CTA computes one row's population variance.
#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op040_variance_kernel(const float* x, float* out, int cols) {
  __shared__ float mean;
  float local = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local += x[blockIdx.x * cols + col];
  const float sum = block_sum(local);
  if (threadIdx.x == 0) mean = sum / cols;
  __syncthreads();
  local = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float delta = x[blockIdx.x * cols + col] - mean;
    local += delta * delta;
  }
  const float square_sum = block_sum(local);
  if (threadIdx.x == 0) out[blockIdx.x] = square_sum / cols;
}

}  // namespace

cudaError_t launch_op040_variance(
    const float* x, float* out, int rows, int cols, cudaStream_t stream) {
  CUDA_CHECK(reductions_detail::validate_matrix(rows, cols));
  if (rows == 0) return cudaSuccess;
  op040_variance_kernel<<<rows, kThreads, 0, stream>>>(x, out, cols);
  return cudaGetLastError();
}
}  // namespace gpu_ops
