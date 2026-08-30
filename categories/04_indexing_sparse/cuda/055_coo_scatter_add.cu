#include "../../../shared/cuda_common.cuh"

namespace gpu_ops {
namespace {

// COO collisions require atomic addition.
__global__ void op055_coo_scatter_add_kernel(
    const int* indices, const float* values, float* out, int nnz, int size) {
  const int position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= nnz) return;
  const int index = indices[position];
  if (index >= 0 && index < size) atomicAdd(out + index, values[position]);
}

}  // namespace

cudaError_t launch_op055_coo_scatter_add(
    const int* indices, const float* values, float* out,
    int nnz, int size, cudaStream_t stream) {
  if (nnz < 0 || size < 0) return cudaErrorInvalidValue;
  // No valid destination exists when size is zero. Avoid launching with an
  // empty allocation even though every lane would be masked by the kernel.
  if (nnz == 0 || size == 0) return cudaSuccess;
  op055_coo_scatter_add_kernel
      <<<ceil_div_int(nnz, kThreads), kThreads, 0, stream>>>(
          indices, values, out, nnz, size);
  return cudaGetLastError();
}

}  // namespace gpu_ops
