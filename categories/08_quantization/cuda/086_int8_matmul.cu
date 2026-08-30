#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op086_int8_matmul_kernel(
    const signed char* a, const signed char* b, const float* a_scale,
    const float* b_scale, float* out, int m, int n, int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;
  int accumulator = 0;
  for (int inner = 0; inner < k; ++inner)
    accumulator += static_cast<int>(a[row * k + inner]) *
                   static_cast<int>(b[inner * n + col]);
  out[row * n + col] = static_cast<float>(accumulator) *
                       a_scale[0] * b_scale[0];
}
}  // namespace

cudaError_t launch_op086_int8_matmul(
    const signed char* a, const signed char* b, const float* a_scale,
    const float* b_scale, float* out, int m, int n, int k,
    cudaStream_t stream) {
  if (m < 0 || n < 0 || k <= 0 ||
      k > quantization_detail::kMaxSafeInt8DotK)
    return cudaErrorInvalidValue;
  if (m == 0 || n == 0) return cudaSuccess;
  const dim3 block(16, 16);
  const dim3 grid(ceil_div_int(n, 16), ceil_div_int(m, 16));
  op086_int8_matmul_kernel<<<grid, block, 0, stream>>>(
      a, b, a_scale, b_scale, out, m, n, k);
  return cudaGetLastError();
}
}  // namespace gpu_ops
