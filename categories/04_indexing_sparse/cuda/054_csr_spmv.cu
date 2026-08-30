#include "../../../shared/cuda_common.cuh"

namespace gpu_ops {
namespace {

// One CTA handles a variable-length CSR row and reduces partial sums.
__global__ void op054_csr_spmv_kernel(
    const int* row_ptr, const int* col_idx, const float* values,
    const float* vector, float* out, int rows, int vector_size) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  float partial = 0.0f;
  for (int position = row_ptr[row] + threadIdx.x;
       position < row_ptr[row + 1]; position += blockDim.x) {
    const int col = col_idx[position];
    if (col >= 0 && col < vector_size)
      partial += values[position] * vector[col];
  }
  const float sum = block_sum(partial);
  if (threadIdx.x == 0) out[row] = sum;
}

}  // namespace

cudaError_t launch_op054_csr_spmv(
    const int* row_ptr, const int* col_idx, const float* values,
    const float* vector, float* out, int rows, int vector_size,
    cudaStream_t stream) {
  if (rows < 0 || vector_size < 0) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  op054_csr_spmv_kernel<<<rows, kThreads, 0, stream>>>(
      row_ptr, col_idx, values, vector, out, rows, vector_size);
  return cudaGetLastError();
}

}  // namespace gpu_ops
