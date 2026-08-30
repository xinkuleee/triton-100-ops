// 042 LogSoftmax: one CTA fuses row maximum, exponential sum, and output.
#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op042_log_softmax_kernel(
    const float* x, float* out, int cols) {
  __shared__ float maximum;
  __shared__ float log_denominator;
  float local_max = -CUDART_INF_F;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local_max = fmaxf(local_max, x[blockIdx.x * cols + col]);
  const float row_max = block_max(local_max);
  if (threadIdx.x == 0) maximum = row_max;
  __syncthreads();
  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local_sum += expf(x[blockIdx.x * cols + col] - maximum);
  const float row_sum = block_sum(local_sum);
  if (threadIdx.x == 0) log_denominator = logf(row_sum) + maximum;
  __syncthreads();
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    out[blockIdx.x * cols + col] =
        x[blockIdx.x * cols + col] - log_denominator;
}

}  // namespace

cudaError_t launch_op042_log_softmax(
    const float* x, float* out, int rows, int cols, cudaStream_t stream) {
  CUDA_CHECK(reductions_detail::validate_matrix(rows, cols));
  if (rows == 0) return cudaSuccess;
  op042_log_softmax_kernel<<<rows, kThreads, 0, stream>>>(x, out, cols);
  return cudaGetLastError();
}
}  // namespace gpu_ops
