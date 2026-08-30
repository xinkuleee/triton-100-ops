#include "../../../shared/cuda_common.cuh"

namespace gpu_ops {
namespace {

__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t bits) {
  // E2M1 finite magnitudes: 0, .5, 1, 1.5, 2, 3, 4, 6.
  constexpr float magnitude[8] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float value = magnitude[bits & 0x7u];
  return (bits & 0x8u) != 0 ? -value : value;
}

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t bits) {
  const int sign = (bits & 0x80u) != 0 ? -1 : 1;
  const int exponent = (bits >> 3) & 0x0f;
  const int mantissa = bits & 0x07;
  if (exponent == 0)
    return sign * ldexpf(static_cast<float>(mantissa), -9);
  if (exponent == 15 && mantissa == 7) return NAN;
  return sign * ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f,
                       exponent - 7);
}

// Triton's Blackwell tutorial stores scales as
// [outer/128, groups/4, 32, 4, 4], then logically transposes the middle
// dimensions to expose [outer, groups].  layout=0 is ordinary row-major;
// layout=1 performs that exact logical-to-packed address conversion.
__device__ __forceinline__ int block_scale_offset(
    int outer_index, int group, int groups, int layout) {
  if (layout == 0) return outer_index * groups + group;
  const int chunk_k_count = groups / 4;
  const int chunk_outer = outer_index / 128;
  const int within_outer = outer_index % 128;
  const int lane = within_outer % 32;
  const int quartet = within_outer / 32;
  const int chunk_k = group / 4;
  const int within_k = group % 4;
  return ((((chunk_outer * chunk_k_count + chunk_k) * 32 + lane) * 4 +
           quartet) *
              4 +
          within_k);
}

// 010: portable mathematical reference. Native Blackwell tcgen05 MMA consumes
// packed FP4/FP8 and pre-shuffled scales and is intentionally not impersonated.
__global__ void op010_block_scaled_matmul_kernel(
    const float* a, const float* b, const float* scale_a,
    const float* scale_b, float* c, int m, int n, int k, int vec) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  const int groups = ceil_div_int(k, vec);
  float accumulator = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    const int group = inner / vec;
    accumulator +=
        (a[row * k + inner] * scale_a[row * groups + group]) *
        (b[inner * n + col] * scale_b[col * groups + group]);
  }
  c[row * n + col] = accumulator;
}

// Packed software references for the tutorial's e2m1 and e4m3 inputs.  B is
// supplied as logical [N, K] (the transposed/column-major operand expected by
// the native path), and scales may be row-major or tutorial-shuffled.
__global__ void op010_packed_fp4_block_scaled_matmul_kernel(
    const uint8_t* packed_a, const uint8_t* packed_b,
    const float* scale_a, const float* scale_b, float* c, int m, int n,
    int k, int vec, int scale_layout) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  const int groups = ceil_div_int(k, vec);
  float accumulator = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    const uint8_t a_byte = packed_a[row * ceil_div_int(k, 2) + inner / 2];
    const uint8_t b_byte = packed_b[col * ceil_div_int(k, 2) + inner / 2];
    const uint8_t a_bits =
        (inner & 1) != 0 ? (a_byte >> 4) : (a_byte & 0x0f);
    const uint8_t b_bits =
        (inner & 1) != 0 ? (b_byte >> 4) : (b_byte & 0x0f);
    const int group = inner / vec;
    const float sa = scale_a[
        block_scale_offset(row, group, groups, scale_layout)];
    const float sb = scale_b[
        block_scale_offset(col, group, groups, scale_layout)];
    accumulator +=
        (fp4_e2m1_to_float(a_bits) * sa) *
        (fp4_e2m1_to_float(b_bits) * sb);
  }
  c[row * n + col] = accumulator;
}

__global__ void op010_packed_fp8_block_scaled_matmul_kernel(
    const uint8_t* packed_a, const uint8_t* packed_b,
    const float* scale_a, const float* scale_b, float* c, int m, int n,
    int k, int vec, int scale_layout) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  const int groups = ceil_div_int(k, vec);
  float accumulator = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    const int group = inner / vec;
    const float sa = scale_a[
        block_scale_offset(row, group, groups, scale_layout)];
    const float sb = scale_b[
        block_scale_offset(col, group, groups, scale_layout)];
    accumulator +=
        (fp8_e4m3fn_to_float(packed_a[row * k + inner]) * sa) *
        (fp8_e4m3fn_to_float(packed_b[col * k + inner]) * sb);
  }
  c[row * n + col] = accumulator;
}

// Software counterpart of the tutorial's mixed path: FP8 E4M3 A multiplied
// by packed FP4 E2M1 B.  Scales are already decoded to float, as in the two
// homogeneous reference launchers above.
__global__ void op010_packed_mixed_block_scaled_matmul_kernel(
    const uint8_t* fp8_a, const uint8_t* packed_fp4_b,
    const float* scale_a, const float* scale_b, float* c, int m, int n,
    int k, int vec, int scale_layout) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  const int groups = ceil_div_int(k, vec);
  const int packed_k = ceil_div_int(k, 2);
  float accumulator = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    const uint8_t b_byte = packed_fp4_b[col * packed_k + inner / 2];
    const uint8_t b_bits =
        (inner & 1) != 0 ? (b_byte >> 4) : (b_byte & 0x0f);
    const int group = inner / vec;
    const float sa = scale_a[
        block_scale_offset(row, group, groups, scale_layout)];
    const float sb = scale_b[
        block_scale_offset(col, group, groups, scale_layout)];
    accumulator +=
        (fp8_e4m3fn_to_float(fp8_a[row * k + inner]) * sa) *
        (fp4_e2m1_to_float(b_bits) * sb);
  }
  c[row * n + col] = accumulator;
}

}  // namespace

cudaError_t launch_op010_block_scaled_matmul(
    const float* a, const float* b, const float* scale_a,
    const float* scale_b, float* c, int m, int n, int k, int vec,
    cudaStream_t stream) {
  if (m < 0 || n < 0 || k < 0 || vec <= 0) return cudaErrorInvalidValue;
  if (m == 0 || n == 0) return cudaSuccess;
  if (c == nullptr || scale_a == nullptr || scale_b == nullptr ||
      (k != 0 && (a == nullptr || b == nullptr)))
    return cudaErrorInvalidValue;
  const dim3 block(16, 16);
  const dim3 grid(ceil_div_int(n, 16), ceil_div_int(m, 16));
  op010_block_scaled_matmul_kernel<<<grid, block, 0, stream>>>(
      a, b, scale_a, scale_b, c, m, n, k, vec);
  return cudaGetLastError();
}

cudaError_t launch_op010_packed_fp4_block_scaled_matmul_reference(
    const uint8_t* packed_a, const uint8_t* packed_b,
    const float* scale_a, const float* scale_b, float* c, int m, int n,
    int k, int vec, int scale_layout, cudaStream_t stream) {
  if (m < 0 || n < 0 || k < 0 || vec <= 0 ||
      (scale_layout != 0 && scale_layout != 1))
    return cudaErrorInvalidValue;
  if (m == 0 || n == 0) return cudaSuccess;
  if (c == nullptr || scale_a == nullptr || scale_b == nullptr ||
      (k != 0 && (packed_a == nullptr || packed_b == nullptr)))
    return cudaErrorInvalidValue;
  const int groups = ceil_div_int(k, vec);
  if (scale_layout == 1 &&
      ((m % 128) != 0 || (n % 128) != 0 || (groups % 4) != 0))
    return cudaErrorInvalidValue;
  const dim3 block(16, 16);
  const dim3 grid(ceil_div_int(n, 16), ceil_div_int(m, 16));
  op010_packed_fp4_block_scaled_matmul_kernel<<<grid, block, 0, stream>>>(
      packed_a, packed_b, scale_a, scale_b, c, m, n, k, vec,
      scale_layout);
  return cudaGetLastError();
}

cudaError_t launch_op010_packed_fp8_block_scaled_matmul_reference(
    const uint8_t* packed_a, const uint8_t* packed_b,
    const float* scale_a, const float* scale_b, float* c, int m, int n,
    int k, int vec, int scale_layout, cudaStream_t stream) {
  if (m < 0 || n < 0 || k < 0 || vec <= 0 ||
      (scale_layout != 0 && scale_layout != 1))
    return cudaErrorInvalidValue;
  if (m == 0 || n == 0) return cudaSuccess;
  if (c == nullptr || scale_a == nullptr || scale_b == nullptr ||
      (k != 0 && (packed_a == nullptr || packed_b == nullptr)))
    return cudaErrorInvalidValue;
  const int groups = ceil_div_int(k, vec);
  if (scale_layout == 1 &&
      ((m % 128) != 0 || (n % 128) != 0 || (groups % 4) != 0))
    return cudaErrorInvalidValue;
  const dim3 block(16, 16);
  const dim3 grid(ceil_div_int(n, 16), ceil_div_int(m, 16));
  op010_packed_fp8_block_scaled_matmul_kernel<<<grid, block, 0, stream>>>(
      packed_a, packed_b, scale_a, scale_b, c, m, n, k, vec,
      scale_layout);
  return cudaGetLastError();
}

cudaError_t launch_op010_packed_mixed_block_scaled_matmul_reference(
    const uint8_t* fp8_a, const uint8_t* packed_fp4_b,
    const float* scale_a, const float* scale_b, float* c, int m, int n,
    int k, int vec, int scale_layout, cudaStream_t stream) {
  if (m < 0 || n < 0 || k < 0 || vec <= 0 ||
      (scale_layout != 0 && scale_layout != 1))
    return cudaErrorInvalidValue;
  if (m == 0 || n == 0) return cudaSuccess;
  if (c == nullptr || scale_a == nullptr || scale_b == nullptr ||
      (k != 0 && (fp8_a == nullptr || packed_fp4_b == nullptr)))
    return cudaErrorInvalidValue;
  const int groups = ceil_div_int(k, vec);
  if (scale_layout == 1 &&
      ((m % 128) != 0 || (n % 128) != 0 || (groups % 4) != 0))
    return cudaErrorInvalidValue;
  const dim3 block(16, 16);
  const dim3 grid(ceil_div_int(n, 16), ceil_div_int(m, 16));
  op010_packed_mixed_block_scaled_matmul_kernel<<<grid, block, 0, stream>>>(
      fp8_a, packed_fp4_b, scale_a, scale_b, c, m, n, k, vec,
      scale_layout);
  return cudaGetLastError();
}

}  // namespace gpu_ops
