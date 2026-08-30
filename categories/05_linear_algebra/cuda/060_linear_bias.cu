#include "_common.cuh"

namespace gpu_ops {
namespace {

// Direct teaching baseline. Bias is fused into the sole global store.
__global__ void op060_linear_bias_kernel(
    const float* x, const float* weight, const float* bias, float* out,
    int m, int n, int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;
  float accumulator = bias[col];
  for (int inner = 0; inner < k; ++inner)
    accumulator += x[row * k + inner] * weight[col * k + inner];
  out[row * n + col] = accumulator;
}

}  // namespace

cudaError_t launch_op060_linear_bias(
    const float* x, const float* weight, const float* bias, float* out,
    int m, int n, int k, cudaStream_t stream) {
  if (k > linear_algebra_detail::kMaxDotK ||
      !linear_algebra_detail::matrix_dims_valid(m, n, k))
    return cudaErrorInvalidValue;
  if (m == 0 || n == 0) return cudaSuccess;
  dim3 block(linear_algebra_detail::kTile, linear_algebra_detail::kTile);
  dim3 grid(ceil_div_int(n, linear_algebra_detail::kTile),
            ceil_div_int(m, linear_algebra_detail::kTile));
  op060_linear_bias_kernel<<<grid, block, 0, stream>>>(
      x, weight, bias, out, m, n, k);
  return cudaGetLastError();
}

}  // namespace gpu_ops
