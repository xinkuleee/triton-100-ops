// 046 Cumsum: one CTA performs an inclusive row scan tile by tile.
#include "_common.cuh"

namespace gpu_ops {
namespace {

// A row owns one CTA.  Each pass scans one 256-element tile with a
// shared-memory Hillis-Steele inclusive scan, then adds the previous tile's
// carry.  This serializes tiles within a row but needs no inter-CTA protocol.
__global__ void op046_cumsum_kernel(
    const float* x, float* out, int cols) {
  __shared__ float scan[kThreads];
  __shared__ float carry;
  if (threadIdx.x == 0) carry = 0.0f;
  __syncthreads();
  const int row_base = blockIdx.x * cols;
  for (int tile = 0; tile < cols; tile += blockDim.x) {
    const int col = tile + threadIdx.x;
    scan[threadIdx.x] = col < cols ? x[row_base + col] : 0.0f;
    __syncthreads();
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
      const float left = threadIdx.x >= offset
          ? scan[threadIdx.x - offset] : 0.0f;
      __syncthreads();
      scan[threadIdx.x] += left;
      __syncthreads();
    }
    const float tile_carry = carry;
    // Every warp must snapshot the old carry before warp 7 updates it.
    __syncthreads();
    if (col < cols) out[row_base + col] = scan[threadIdx.x] + tile_carry;
    if (threadIdx.x == blockDim.x - 1) carry += scan[threadIdx.x];
    __syncthreads();
  }
}

}  // namespace

cudaError_t launch_op046_cumsum(
    const float* x, float* out, int rows, int cols, cudaStream_t stream) {
  CUDA_CHECK(reductions_detail::validate_matrix(rows, cols));
  if (rows == 0) return cudaSuccess;
  op046_cumsum_kernel<<<rows, kThreads, 0, stream>>>(x, out, cols);
  return cudaGetLastError();
}
}  // namespace gpu_ops
