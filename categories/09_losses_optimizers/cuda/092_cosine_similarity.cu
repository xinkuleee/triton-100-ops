#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op092_cosine_similarity_kernel(
    const float* a, const float* b, float* out, int cols, float eps) {
  const int row = blockIdx.x;
  float local_dot = 0.0f;
  float local_a = 0.0f;
  float local_b = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float x = a[row * cols + col];
    const float y = b[row * cols + col];
    local_dot += x * y;
    local_a += x * x;
    local_b += y * y;
  }
  __shared__ float reductions[3];
  const float dot = block_sum(local_dot);
  if (threadIdx.x == 0) reductions[0] = dot;
  __syncthreads();
  const float squared_a = block_sum(local_a);
  if (threadIdx.x == 0) reductions[1] = squared_a;
  __syncthreads();
  const float squared_b = block_sum(local_b);
  if (threadIdx.x == 0) reductions[2] = squared_b;
  __syncthreads();
  if (threadIdx.x == 0)
    out[row] = reductions[0] /
               (fmaxf(sqrtf(reductions[1]), eps) *
                fmaxf(sqrtf(reductions[2]), eps));
}

}  // namespace

cudaError_t launch_op092_cosine_similarity(
    const float* a, const float* b, float* out, int rows, int cols, float eps,
    cudaStream_t stream) {
  if (rows < 0 || cols <= 0 || !std::isfinite(eps) || eps <= 0.0f)
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  op092_cosine_similarity_kernel<<<rows, kThreads, 0, stream>>>(
      a, b, out, cols, eps);
  return cudaGetLastError();
}

}  // namespace gpu_ops
