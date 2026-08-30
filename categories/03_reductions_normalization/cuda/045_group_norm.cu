// 045 GroupNorm: one CTA computes and applies statistics for one batch-group.
#include "_common.cuh"

namespace gpu_ops {
namespace {

__global__ void op045_group_norm_kernel(
    const float* x, const float* gamma, const float* beta, float* out,
    int channels, int spatial, int groups, float eps) {
  __shared__ float mean;
  __shared__ float inverse_std;
  const int channels_per_group = channels / groups;
  const int count = channels_per_group * spatial;
  const int group = blockIdx.x % groups;
  const int batch = blockIdx.x / groups;
  const int base = batch * channels * spatial + group * count;
  float local = 0.0f;
  for (int index = threadIdx.x; index < count; index += blockDim.x)
    local += x[base + index];
  const float sum = block_sum(local);
  if (threadIdx.x == 0) mean = sum / count;
  __syncthreads();
  local = 0.0f;
  for (int index = threadIdx.x; index < count; index += blockDim.x) {
    const float delta = x[base + index] - mean;
    local += delta * delta;
  }
  const float square_sum = block_sum(local);
  if (threadIdx.x == 0) inverse_std = rsqrtf(square_sum / count + eps);
  __syncthreads();
  for (int index = threadIdx.x; index < count; index += blockDim.x) {
    const int channel = group * channels_per_group + index / spatial;
    out[base + index] = (x[base + index] - mean) * inverse_std *
                        gamma[channel] + beta[channel];
  }
}

}  // namespace

cudaError_t launch_op045_group_norm(
    const float* x, const float* gamma, const float* beta, float* out,
    int batch, int channels, int spatial, int groups, float eps,
    cudaStream_t stream) {
  if (batch < 0 || channels <= 0 || spatial <= 0 || groups <= 0 ||
      channels % groups != 0 || !std::isfinite(eps) || eps < 0.0f)
    return cudaErrorInvalidValue;
  if (batch == 0) return cudaSuccess;
  op045_group_norm_kernel<<<batch * groups, kThreads, 0, stream>>>(
      x, gamma, beta, out, channels, spatial, groups, eps);
  return cudaGetLastError();
}
}  // namespace gpu_ops
