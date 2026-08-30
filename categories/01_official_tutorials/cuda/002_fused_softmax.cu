#include "../../../shared/cuda_common.cuh"

#include <cfloat>

namespace gpu_ops {
namespace {

// 002: each persistent CTA owns several rows through a grid-stride loop.
__global__ void op002_fused_softmax_kernel(
    const float* x, float* out, int rows, int cols, int input_row_stride,
    int output_row_stride) {
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const size_t input_base =
        static_cast<size_t>(row) * input_row_stride;
    const size_t output_base =
        static_cast<size_t>(row) * output_row_stride;
    float local_max = -FLT_MAX;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
      local_max = fmaxf(local_max, x[input_base + col]);
    const float maximum = block_max(local_max);

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
      local_sum += expf(x[input_base + col] - maximum);
    const float denominator = block_sum(local_sum);

    for (int col = threadIdx.x; col < cols; col += blockDim.x)
      out[output_base + col] =
          expf(x[input_base + col] - maximum) / denominator;
    __syncthreads();  // keep the next grid-stride row from racing shared partials
  }
}

}  // namespace

cudaError_t launch_op002_fused_softmax_strided(
    const float* x, float* out, int rows, int cols, int input_row_stride,
    int output_row_stride, int persistent_ctas, cudaStream_t stream);

cudaError_t launch_op002_fused_softmax(
    const float* x, float* out, int rows, int cols, int persistent_ctas,
    cudaStream_t stream) {
  return launch_op002_fused_softmax_strided(
      x, out, rows, cols, cols, cols, persistent_ctas, stream);
}

// Strided counterpart of the official input_row_stride/output_row_stride API.
cudaError_t launch_op002_fused_softmax_strided(
    const float* x, float* out, int rows, int cols, int input_row_stride,
    int output_row_stride, int persistent_ctas, cudaStream_t stream) {
  if (rows < 0 || cols <= 0 || persistent_ctas <= 0)
    return cudaErrorInvalidValue;
  if (input_row_stride < cols || output_row_stride < cols)
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (x == nullptr || out == nullptr) return cudaErrorInvalidValue;
  const int grid = rows < persistent_ctas ? rows : persistent_ctas;
  op002_fused_softmax_kernel<<<grid, kThreads, 0, stream>>>(
      x, out, rows, cols, input_row_stride, output_row_stride);
  return cudaGetLastError();
}

}  // namespace gpu_ops
