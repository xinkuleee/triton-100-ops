#include "../../../shared/cuda_common.cuh"

#include <climits>

namespace gpu_ops {
namespace {

// 003: a portable FP32 CUDA teaching path. A CTA computes one shared-memory
// tile, and blockIdx.x is remapped in groups of M tiles so nearby CTAs reuse B
// tiles through L2. The official FP16/FP8 tl.dot tensor-core/autotuned path is
// a hardware-specialized extension, not something this kernel impersonates.
__global__ void op003_matrix_multiplication_kernel(
    const float* a, const float* b, float* c, int m, int n, int k,
    int lda, int ldb, int ldc, int group_m, bool fused_leaky_relu) {
  constexpr int kTile = 16;
  __shared__ float a_tile[kTile][kTile];
  __shared__ float b_tile[kTile][kTile];
  const int tiles_m = ceil_div_int(m, kTile);
  const int tiles_n = ceil_div_int(n, kTile);
  const int programs_per_group = group_m * tiles_n;
  const int group = blockIdx.x / programs_per_group;
  const int first_m = group * group_m;
  const int actual_m = min(group_m, tiles_m - first_m);
  const int local = blockIdx.x % programs_per_group;
  const int tile_m = first_m + local % actual_m;
  const int tile_n = local / actual_m;
  const int row = tile_m * kTile + threadIdx.y;
  const int col = tile_n * kTile + threadIdx.x;
  float accumulator = 0.0f;
  for (int k0 = 0; k0 < k; k0 += kTile) {
    const int a_col = k0 + threadIdx.x;
    const int b_row = k0 + threadIdx.y;
    a_tile[threadIdx.y][threadIdx.x] =
        row < m && a_col < k ? a[static_cast<size_t>(row) * lda + a_col]
                             : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        b_row < k && col < n ? b[static_cast<size_t>(b_row) * ldb + col]
                             : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int inner = 0; inner < kTile; ++inner)
      accumulator += a_tile[threadIdx.y][inner] *
                     b_tile[inner][threadIdx.x];
    __syncthreads();
  }
  if (row < m && col < n) {
    // Mirrors the official ACTIVATION constexpr epilogue.  It remains in the
    // same kernel, so the FP32 accumulator never makes a round trip through HBM.
    if (fused_leaky_relu && accumulator < 0.0f) accumulator *= 0.01f;
    c[static_cast<size_t>(row) * ldc + col] = accumulator;
  }
}

}  // namespace

cudaError_t launch_op003_matrix_multiplication_strided(
    const float* a, const float* b, float* c, int m, int n, int k, int lda,
    int ldb, int ldc, int group_m, bool fused_leaky_relu,
    cudaStream_t stream);

cudaError_t launch_op003_matrix_multiplication(
    const float* a, const float* b, float* c, int m, int n, int k,
    int group_m, cudaStream_t stream) {
  return launch_op003_matrix_multiplication_strided(
      a, b, c, m, n, k, k, n, n, group_m, false, stream);
}

cudaError_t launch_op003_matrix_multiplication_leaky_relu(
    const float* a, const float* b, float* c, int m, int n, int k,
    int group_m, cudaStream_t stream) {
  return launch_op003_matrix_multiplication_strided(
      a, b, c, m, n, k, k, n, n, group_m, true, stream);
}

// General row-strided launch, matching the pointer arithmetic taught by the
// Triton kernel.  Set fused_leaky_relu to keep the activation in the epilogue.
cudaError_t launch_op003_matrix_multiplication_strided(
    const float* a, const float* b, float* c, int m, int n, int k, int lda,
    int ldb, int ldc, int group_m, bool fused_leaky_relu,
    cudaStream_t stream) {
  if (m < 0 || n < 0 || k < 0 || lda < k || ldb < n || ldc < n ||
      group_m <= 0)
    return cudaErrorInvalidValue;
  if (m == 0 || n == 0) return cudaSuccess;
  if (c == nullptr || (k != 0 && (a == nullptr || b == nullptr)))
    return cudaErrorInvalidValue;
  constexpr int kTile = 16;
  const int tiles_m = ceil_div_int(m, kTile);
  const int tiles_n = ceil_div_int(n, kTile);
  if (tiles_n != 0 && tiles_m > INT_MAX / tiles_n)
    return cudaErrorInvalidValue;
  const int blocks = tiles_m * tiles_n;
  const int effective_group_m = group_m < tiles_m ? group_m : tiles_m;
  op003_matrix_multiplication_kernel<<<blocks, dim3(kTile, kTile), 0, stream>>>(
      a, b, c, m, n, k, lda, ldb, ldc, effective_group_m,
      fused_leaky_relu);
  return cudaGetLastError();
}

}  // namespace gpu_ops
