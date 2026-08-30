#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op061_conv1d_kernel(
    const float* x, const float* weight, const float* bias, float* out,
    int total, int channels, int length, int out_channels, int kernel_size,
    int out_length, int stride, int padding) {
  const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= total) return;
  const int output_x = output_index % out_length;
  const int output_channel = (output_index / out_length) % out_channels;
  const int batch = output_index / (out_channels * out_length);
  float accumulator = bias[output_channel];
  for (int channel = 0; channel < channels; ++channel)
    for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x) {
      const int input_x = output_x * stride + kernel_x - padding;
      if (input_x >= 0 && input_x < length)
        accumulator +=
            x[(batch * channels + channel) * length + input_x] *
            weight[(output_channel * channels + channel) * kernel_size +
                   kernel_x];
    }
  out[output_index] = accumulator;
}
}  // namespace

cudaError_t launch_op061_conv1d(
    const float* x, const float* weight, const float* bias, float* out,
    int batch, int channels, int length, int out_channels, int kernel_size,
    int stride, int padding, cudaStream_t stream) {
  if (batch < 0 || channels <= 0 || length <= 0 || out_channels <= 0 ||
      kernel_size <= 0 || stride <= 0 || padding < 0)
    return cudaErrorInvalidValue;
  if (length + 2 * padding < kernel_size) return cudaSuccess;
  const int out_length = (length + 2 * padding - kernel_size) / stride + 1;
  const long long wide_total = 1LL * batch * out_channels * out_length;
  if (!vision_detail::total_fits(wide_total)) return cudaErrorInvalidValue;
  const int total = static_cast<int>(wide_total);
  if (total == 0) return cudaSuccess;
  op061_conv1d_kernel<<<ceil_div_int(total, kThreads), kThreads, 0, stream>>>(
      x, weight, bias, out, total, channels, length, out_channels,
      kernel_size, out_length, stride, padding);
  return cudaGetLastError();
}
}  // namespace gpu_ops
