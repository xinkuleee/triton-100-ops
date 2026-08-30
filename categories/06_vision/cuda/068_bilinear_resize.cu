#include "_common.cuh"

namespace gpu_ops {
namespace {
__global__ void op068_bilinear_resize_kernel(
    const float* x, float* out, int total, int channels,
    int input_height, int input_width, int out_height, int out_width) {
  const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= total) return;
  const int output_x = output_index % out_width;
  const int output_y = (output_index / out_width) % out_height;
  const int channel = (output_index / (out_width * out_height)) % channels;
  const int batch = output_index / (channels * out_height * out_width);
  const float source_y =
      fmaxf((output_y + 0.5f) * input_height / out_height - 0.5f, 0.0f);
  const float source_x =
      fmaxf((output_x + 0.5f) * input_width / out_width - 0.5f, 0.0f);
  const int y0 = static_cast<int>(source_y);
  const int x0 = static_cast<int>(source_x);
  const int y1 = min(y0 + 1, input_height - 1);
  const int x1 = min(x0 + 1, input_width - 1);
  const float wy = source_y - y0;
  const float wx = source_x - x0;
  const float* image =
      x + (batch * channels + channel) * input_height * input_width;
  const float top =
      (1.0f - wx) * image[y0 * input_width + x0] +
      wx * image[y0 * input_width + x1];
  const float bottom =
      (1.0f - wx) * image[y1 * input_width + x0] +
      wx * image[y1 * input_width + x1];
  out[output_index] = (1.0f - wy) * top + wy * bottom;
}
}  // namespace

cudaError_t launch_op068_bilinear_resize(
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
  op068_bilinear_resize_kernel<<<
      ceil_div_int(total, kThreads), kThreads, 0, stream>>>(
      x, out, total, channels, input_height, input_width, out_height,
      out_width);
  return cudaGetLastError();
}
}  // namespace gpu_ops
