// 038 L1 Norm: one CTA reduces the absolute values in one FP32 row.
#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op038_l1_norm_kernel(const float* x, float* out, int cols) {
  float local = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local += fabsf(x[blockIdx.x * cols + col]);
  const float result = block_sum(local);
  if (threadIdx.x == 0) out[blockIdx.x] = result;
}
}  // namespace

cudaError_t launch_op038_l1_norm(const float* x, float* out, int rows, int cols, cudaStream_t stream) {
  CUDA_CHECK(reductions_detail::validate_matrix(rows, cols));
  if (rows == 0) return cudaSuccess;
  op038_l1_norm_kernel<<<rows, kThreads, 0, stream>>>(x, out, cols);
  return cudaGetLastError();
}
}  // namespace gpu_ops
