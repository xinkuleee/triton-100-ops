#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op091_cross_entropy_kernel(
    const float* logits, const int* labels, float* out, int cols) {
  const int row = blockIdx.x;
  float local_maximum = -FLT_MAX;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local_maximum = fmaxf(local_maximum, logits[row * cols + col]);
  __shared__ float saved_maximum;
  const float maximum_value = block_max(local_maximum);
  if (threadIdx.x == 0) saved_maximum = maximum_value;
  __syncthreads();
  const float maximum = saved_maximum;
  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local_sum += expf(logits[row * cols + col] - maximum);
  const float sum = block_sum(local_sum);
  if (threadIdx.x == 0) {
    const int label = labels[row];
    out[row] = label >= 0 && label < cols
        ? maximum + logf(sum) - logits[row * cols + label]
        : CUDART_NAN_F;
  }
}

}  // namespace

cudaError_t launch_op091_cross_entropy(
    const float* logits, const int* labels, float* out, int rows, int cols,
    cudaStream_t stream) {
  if (rows < 0 || cols <= 0) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  op091_cross_entropy_kernel<<<rows, kThreads, 0, stream>>>(
      logits, labels, out, cols);
  return cudaGetLastError();
}

}  // namespace gpu_ops
