#include "_common.cuh"

namespace gpu_ops {
__global__ void op065_max_pool2d_kernel(
    const float* x, float* out, int total, int height, int width,
    int out_height, int out_width, int kernel_height, int kernel_width,
    int stride, int padding) {
  const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= total) return;
  const int output_x = output_index % out_width;
  const int output_y = (output_index / out_width) % out_height;
  const int batch_channel = output_index / (out_width * out_height);
  float maximum = -INFINITY;
  for (int kernel_y = 0; kernel_y < kernel_height; ++kernel_y)
    for (int kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
      const int input_y = output_y * stride + kernel_y - padding;
      const int input_x = output_x * stride + kernel_x - padding;
      if (input_y >= 0 && input_y < height && input_x >= 0 && input_x < width)
        maximum = fmaxf(maximum, x[(batch_channel * height + input_y) * width + input_x]);
    }
  out[output_index] = maximum;
}

cudaError_t launch_op065_max_pool2d(
    const float* x, float* out, int batch, int channels, int height, int width,
    int kernel_height, int kernel_width, int stride, int padding,
    cudaStream_t stream) {
  if (batch < 0 || channels < 0 || !vision_detail::spatial_valid(height, width, kernel_height, kernel_width, stride, padding)) return cudaErrorInvalidValue;
  if (height + 2 * padding < kernel_height || width + 2 * padding < kernel_width) return cudaSuccess;
  const int out_height = (height + 2 * padding - kernel_height) / stride + 1;
  const int out_width = (width + 2 * padding - kernel_width) / stride + 1;
  const long long wide_total = 1LL * batch * channels * out_height * out_width;
  if (!vision_detail::total_fits(wide_total)) return cudaErrorInvalidValue;
  const int total = static_cast<int>(wide_total);
  if (total == 0) return cudaSuccess;
  op065_max_pool2d_kernel<<<ceil_div_int(total, kThreads), kThreads, 0, stream>>>(x, out, total, height, width, out_height, out_width, kernel_height, kernel_width, stride, padding);
  return cudaGetLastError();
}
}  // namespace gpu_ops
