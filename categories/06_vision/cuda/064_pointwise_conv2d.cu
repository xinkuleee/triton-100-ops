#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op064_pointwise_conv2d_kernel(
    const float* x, const float* weight, const float* bias, float* out,
    int total, int channels, int out_channels, int pixels) {
  const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= total) return;
  const int pixel = output_index % pixels;
  const int output_channel = (output_index / pixels) % out_channels;
  const int batch = output_index / (pixels * out_channels);
  float accumulator = bias[output_channel];
  for (int channel = 0; channel < channels; ++channel)
    accumulator += x[(batch * channels + channel) * pixels + pixel] *
                   weight[output_channel * channels + channel];
  out[output_index] = accumulator;
}
}  // namespace

cudaError_t launch_op064_pointwise_conv2d(
    const float* x, const float* weight, const float* bias, float* out,
    int batch, int channels, int height, int width, int out_channels,
    cudaStream_t stream) {
  const long long wide_total =
      1LL * batch * out_channels * height * width;
  if (batch < 0 || channels <= 0 || height <= 0 || width <= 0 ||
      out_channels <= 0 || !vision_detail::total_fits(wide_total))
    return cudaErrorInvalidValue;
  const int total = static_cast<int>(wide_total);
  if (total == 0) return cudaSuccess;
  op064_pointwise_conv2d_kernel<<<
      ceil_div_int(total, kThreads), kThreads, 0, stream>>>(
      x, weight, bias, out, total, channels, out_channels, height * width);
  return cudaGetLastError();
}
}  // namespace gpu_ops
