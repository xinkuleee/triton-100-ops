#include "../../../shared/cuda_common.cuh"

namespace gpu_ops {
namespace {

// 008: all heterogeneous GEMM tiles form one logical queue. This is the
// portable pointer-list path; the official Hopper TMA descriptor variant and
// tensor-core datatype path are separate hardware-specialized work.
__global__ void op008_grouped_gemm_kernel(
    const float* const* a, const float* const* b, float* const* c,
    const int* sizes, const int* strides, int groups, int workers) {
  constexpr int kTile = 16;
  __shared__ float a_tile[kTile][kTile];
  __shared__ float b_tile[kTile][kTile];
  int64_t tile = blockIdx.x;
  int64_t problem_begin = 0;
  for (int group = 0; group < groups; ++group) {
    const int m = sizes[3 * group];
    const int n = sizes[3 * group + 1];
    const int k = sizes[3 * group + 2];
    const int lda = strides[3 * group];
    const int ldb = strides[3 * group + 1];
    const int ldc = strides[3 * group + 2];
    const int tiles_m = ceil_div_int(m, kTile);
    const int tiles_n = ceil_div_int(n, kTile);
    const int64_t problem_end =
        problem_begin + static_cast<int64_t>(tiles_m) * tiles_n;
    // Do not restart blockIdx.x for each problem.  The worker advances through
    // one concatenated tile queue, exactly like the tutorial's tile_idx.
    for (; tile >= problem_begin && tile < problem_end; tile += workers) {
      const int local = static_cast<int>(tile - problem_begin);
      const int row = (local / tiles_n) * kTile + threadIdx.y;
      const int col = (local % tiles_n) * kTile + threadIdx.x;
      float accumulator = 0.0f;
      for (int k0 = 0; k0 < k; k0 += kTile) {
        const int a_col = k0 + threadIdx.x;
        const int b_row = k0 + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] =
            row < m && a_col < k ? a[group][row * lda + a_col] : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] =
            b_row < k && col < n ? b[group][b_row * ldb + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int inner = 0; inner < kTile; ++inner)
          accumulator += a_tile[threadIdx.y][inner] *
                         b_tile[inner][threadIdx.x];
        __syncthreads();
      }
      if (row < m && col < n) c[group][row * ldc + col] = accumulator;
      __syncthreads();
    }
    problem_begin = problem_end;
  }
}

}  // namespace

cudaError_t launch_op008_grouped_gemm(
    const float* const* a, const float* const* b, float* const* c,
    const int* sizes, const int* strides, int groups, int workers,
    cudaStream_t stream) {
  if (groups < 0 || workers <= 0) return cudaErrorInvalidValue;
  if (groups == 0) return cudaSuccess;
  if (a == nullptr || b == nullptr || c == nullptr || sizes == nullptr ||
      strides == nullptr)
    return cudaErrorInvalidValue;
  op008_grouped_gemm_kernel<<<workers, dim3(16, 16), 0, stream>>>(
      a, b, c, sizes, strides, groups, workers);
  return cudaGetLastError();
}

}  // namespace gpu_ops
