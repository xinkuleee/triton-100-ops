#include "../../../shared/cuda_common.cuh"

namespace gpu_ops {
namespace {

// 011 -----------------------------------------------------------------------
__global__ void op011_programmatic_dependent_launch_kernel(
    const float* x, const float* y, float* out, int n, bool enable_pdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (enable_pdl) cudaGridDependencySynchronize();
#endif
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  float xv = 0.0f;
  float yv = 0.0f;
  if (index < n) {
    xv = x[index];
    yv = y[index];
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (enable_pdl) cudaTriggerProgrammaticLaunchCompletion();
#endif
  if (index < n) out[index] = xv + yv;
}

}  // namespace

cudaError_t launch_op011_programmatic_dependent_launch(
    const float* x, const float* y, float* out, int n, bool enable_pdl,
    cudaStream_t stream) {
  if (n < 0) return cudaErrorInvalidValue;
  if (n == 0) return cudaSuccess;
  if (x == nullptr || y == nullptr || out == nullptr)
    return cudaErrorInvalidValue;
#if CUDART_VERSION >= 12000
  if (enable_pdl) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int major = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, device));
    if (major < 9) return cudaErrorNotSupported;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(ceil_div_int(n, kThreads));
  config.blockDim = dim3(kThreads);
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0;
  config.attrs = enable_pdl ? &attribute : nullptr;
  config.numAttrs = enable_pdl ? 1 : 0;
  CUDA_CHECK(cudaLaunchKernelEx(
      &config, op011_programmatic_dependent_launch_kernel,
      x, y, out, n, enable_pdl));
  return cudaGetLastError();
#else
  if (enable_pdl) return cudaErrorNotSupported;
  op011_programmatic_dependent_launch_kernel<<<
      ceil_div_int(n, kThreads), kThreads, 0, stream>>>(x, y, out, n, false);
  return cudaGetLastError();
#endif
}

}  // namespace gpu_ops
