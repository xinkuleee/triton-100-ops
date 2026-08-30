#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op100_non_finite_check_kernel(
    const float* x, int* flag, int count) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < count && !isfinite(x[offset])) atomicOr(flag, 1);
}

}  // namespace

cudaError_t launch_op100_non_finite_check(
    const float* x, int* flag, int count, cudaStream_t stream) {
  if (count < 0) return cudaErrorInvalidValue;
  CUDA_CHECK(cudaMemsetAsync(flag, 0, sizeof(int), stream));
  if (count == 0) return cudaSuccess;
  op100_non_finite_check_kernel<<<
      ceil_div_int(count, kThreads), kThreads, 0, stream>>>(x, flag, count);
  return cudaGetLastError();
}

}  // namespace gpu_ops
