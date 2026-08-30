// 039 L2 Norm: one CTA accumulates squared values and takes one square root.
#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op039_l2_norm_kernel(const float* x, float* out, int cols) {
  float local = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = x[blockIdx.x * cols + col];
    local += value * value;
  }
  const float result = block_sum(local);
  if (threadIdx.x == 0) out[blockIdx.x] = sqrtf(result);
}
}  // namespace

cudaError_t launch_op039_l2_norm(const float* x, float* out, int rows, int cols, cudaStream_t stream) {
  CUDA_CHECK(reductions_detail::validate_matrix(rows, cols));
  if (rows == 0) return cudaSuccess;
  op039_l2_norm_kernel<<<rows, kThreads, 0, stream>>>(x, out, cols);
  return cudaGetLastError();
}
}  // namespace gpu_ops
