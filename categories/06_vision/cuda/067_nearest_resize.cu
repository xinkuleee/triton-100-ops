#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op067_nearest_resize_kernel(
    const float* x, float* out, int total, int channels,
    int input_height, int input_width, int out_height, int out_width) {
  const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= total) return;
  const int output_x = output_index % out_width;
  const int output_y = (output_index / out_width) % out_height;
  const int channel = (output_index / (out_width * out_height)) % channels;
  const int batch = output_index / (channels * out_height * out_width);
  const int input_y = (1LL * output_y * input_height) / out_height;
  const int input_x = (1LL * output_x * input_width) / out_width;
  out[output_index] =
      x[((batch * channels + channel) * input_height + input_y) *
            input_width +
        input_x];
}
}  // namespace

cudaError_t launch_op067_nearest_resize(
    const float* x, float* out, int batch, int channels, int input_height,
    int input_width, int out_height, int out_width, cudaStream_t stream) {
  const long long wide_total =
      1LL * batch * channels * out_height * out_width;
  if (batch < 0 || channels <= 0 || input_height <= 0 || input_width <= 0 ||
      out_height <= 0 || out_width <= 0 ||
      !vision_detail::total_fits(wide_total))
    return cudaErrorInvalidValue;
  const int total = static_cast<int>(wide_total);
  if (total == 0) return cudaSuccess;
  op067_nearest_resize_kernel<<<
      ceil_div_int(total, kThreads), kThreads, 0, stream>>>(
      x, out, total, channels, input_height, input_width, out_height,
      out_width);
  return cudaGetLastError();
}
}  // namespace gpu_ops
