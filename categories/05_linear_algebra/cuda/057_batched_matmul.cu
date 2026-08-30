#include "_common.cuh"

namespace gpu_ops {
namespace {

// blockIdx.z selects a batch; a 16x16 CTA cooperatively reuses tiles.
__global__ void op057_batched_matmul_kernel(
    const float* a, const float* b, float* out, int m, int n, int k) {
  __shared__ float a_tile[linear_algebra_detail::kTile]
                         [linear_algebra_detail::kTile];
  __shared__ float b_tile[linear_algebra_detail::kTile]
                         [linear_algebra_detail::kTile];
  const int batch = blockIdx.z;
  const int row = blockIdx.y * linear_algebra_detail::kTile + threadIdx.y;
  const int col = blockIdx.x * linear_algebra_detail::kTile + threadIdx.x;
  const float* batch_a = a + batch * m * k;
  const float* batch_b = b + batch * k * n;
  float accumulator = 0.0f;
  for (int k0 = 0; k0 < k; k0 += linear_algebra_detail::kTile) {
    const int a_col = k0 + threadIdx.x;
    const int b_row = k0 + threadIdx.y;
    a_tile[threadIdx.y][threadIdx.x] =
        row < m && a_col < k ? batch_a[row * k + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        b_row < k && col < n ? batch_b[b_row * n + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int inner = 0; inner < linear_algebra_detail::kTile; ++inner)
      accumulator += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
    __syncthreads();
  }
  if (row < m && col < n)
    out[(batch * m + row) * n + col] = accumulator;
}

}  // namespace

cudaError_t launch_op057_batched_matmul(
    const float* a, const float* b, float* out, int batch, int m, int n, int k,
    cudaStream_t stream) {
  if (batch < 0 || k > linear_algebra_detail::kMaxDotK ||
      !linear_algebra_detail::matrix_dims_valid(m, n, k))
    return cudaErrorInvalidValue;
  if (batch == 0 || m == 0 || n == 0) return cudaSuccess;
  dim3 block(linear_algebra_detail::kTile, linear_algebra_detail::kTile);
  dim3 grid(ceil_div_int(n, linear_algebra_detail::kTile),
            ceil_div_int(m, linear_algebra_detail::kTile), batch);
  op057_batched_matmul_kernel<<<grid, block, 0, stream>>>(a, b, out, m, n, k);
  return cudaGetLastError();
}

}  // namespace gpu_ops
