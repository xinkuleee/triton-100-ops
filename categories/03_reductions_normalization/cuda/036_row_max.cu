// 036 Row Max: one CTA finds the maximum of one contiguous FP32 row.
#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op036_row_max_kernel(const float* x, float* out, int cols) {
  float local = -CUDART_INF_F;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local = fmaxf(local, x[blockIdx.x * cols + col]);
  const float result = block_max(local);
  if (threadIdx.x == 0) out[blockIdx.x] = result;
}
}  // namespace

cudaError_t launch_op036_row_max(const float* x, float* out, int rows, int cols, cudaStream_t stream) {
  CUDA_CHECK(reductions_detail::validate_matrix(rows, cols));
  if (rows == 0) return cudaSuccess;
  op036_row_max_kernel<<<rows, kThreads, 0, stream>>>(x, out, cols);
  return cudaGetLastError();
}
}  // namespace gpu_ops
