#include "../../../shared/cuda_common.cuh"

#include <cstdint>

namespace gpu_ops {
namespace {

// One CTA copies each selected complete row.
__global__ void op051_index_select_rows_kernel(
    const float* x, const int64_t* indices, float* out, int rows, int cols) {
  const int64_t source_row = indices[blockIdx.x];
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    out[blockIdx.x * cols + col] =
        source_row >= 0 && source_row < rows
            ? x[source_row * cols + col]
            : 0.0f;
}

}  // namespace

cudaError_t launch_op051_index_select_rows(
    const float* x, const int64_t* indices, float* out,
    int selected, int rows, int cols, cudaStream_t stream) {
  if (selected < 0 || rows < 0 || cols <= 0) return cudaErrorInvalidValue;
  if (selected == 0) return cudaSuccess;
  op051_index_select_rows_kernel<<<selected, kThreads, 0, stream>>>(
      x, indices, out, rows, cols);
  return cudaGetLastError();
}

}  // namespace gpu_ops
