#include "../../../shared/cuda_common.cuh"

namespace gpu_ops {
namespace {

// 005 -----------------------------------------------------------------------
__global__ void op005_layer_norm_forward_kernel(
    const float* x, const float* weight, const float* bias, float* y,
    float* mean_out, float* rstd_out, int cols, float eps) {
  const int row = blockIdx.x;
  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local_sum += x[row * cols + col];
  __shared__ float saved_mean;
  const float mean_value = block_sum(local_sum) / cols;
  if (threadIdx.x == 0) saved_mean = mean_value;
  __syncthreads();
  const float mean = saved_mean;
  float local_square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = x[row * cols + col] - mean;
    local_square_sum += centered * centered;
  }
  const float rstd = rsqrtf(block_sum(local_square_sum) / cols + eps);
  if (threadIdx.x == 0) {
    mean_out[row] = mean;
    rstd_out[row] = rstd;
  }
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    y[row * cols + col] =
        (x[row * cols + col] - mean) * rstd * weight[col] + bias[col];
}

__global__ void op005_layer_norm_backward_rows_kernel(
    const float* dy, const float* x, const float* weight, const float* mean,
    const float* rstd, float* dx, float* partial_dw, float* partial_db,
    int cols) {
  const int row = blockIdx.x;
  float local_wdy = 0.0f;
  float local_wdy_xhat = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int index = row * cols + col;
    const float xhat = (x[index] - mean[row]) * rstd[row];
    const float wdy = weight[col] * dy[index];
    local_wdy += wdy;
    local_wdy_xhat += wdy * xhat;
  }
  __shared__ float mean_wdy;
  const float reduced_wdy = block_sum(local_wdy);
  if (threadIdx.x == 0) mean_wdy = reduced_wdy / cols;
  __syncthreads();
  const float mean_wdy_xhat = block_sum(local_wdy_xhat) / cols;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int index = row * cols + col;
    const float xhat = (x[index] - mean[row]) * rstd[row];
    const float wdy = weight[col] * dy[index];
    dx[index] = rstd[row] *
        (wdy - mean_wdy - xhat * mean_wdy_xhat);
    partial_dw[index] = dy[index] * xhat;
    partial_db[index] = dy[index];
  }
}

__global__ void op005_layer_norm_backward_params_kernel(
    const float* partial_dw, const float* partial_db, float* dw, float* db,
    int rows, int cols) {
  const int col = blockIdx.x;
  float local_dw = 0.0f;
  float local_db = 0.0f;
  for (int row = threadIdx.x; row < rows; row += blockDim.x) {
    local_dw += partial_dw[row * cols + col];
    local_db += partial_db[row * cols + col];
  }
  __shared__ float reduced_dw;
  const float sum_dw = block_sum(local_dw);
  if (threadIdx.x == 0) reduced_dw = sum_dw;
  __syncthreads();
  const float sum_db = block_sum(local_db);
  if (threadIdx.x == 0) {
    dw[col] = reduced_dw;
    db[col] = sum_db;
  }
}

// Official-style Stage 1.  Rows hash into GROUP_SIZE_M L2-resident buffers.
// A per-buffer lock serializes the vector update, while independent lock IDs
// can progress concurrently.  locks_and_counts must hold 2 * group_size ints:
// locks first, then first-writer counters.
__global__ void op005_layer_norm_backward_grouped_kernel(
    const float* dy, const float* x, const float* weight, const float* mean,
    const float* rstd, float* dx, float* grouped_dw, float* grouped_db,
    int* locks_and_counts, int cols, int group_size) {
  const int row = blockIdx.x;
  const int lock_id = row % group_size;
  float local_wdy = 0.0f;
  float local_wdy_xhat = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int index = row * cols + col;
    const float xhat = (x[index] - mean[row]) * rstd[row];
    const float wdy = weight[col] * dy[index];
    local_wdy += wdy;
    local_wdy_xhat += wdy * xhat;
  }
  __shared__ float mean_wdy;
  __shared__ float mean_wdy_xhat;
  __shared__ int first_writer;
  const float sum_wdy = block_sum(local_wdy);
  if (threadIdx.x == 0) mean_wdy = sum_wdy / cols;
  __syncthreads();
  const float sum_wdy_xhat = block_sum(local_wdy_xhat);
  if (threadIdx.x == 0) mean_wdy_xhat = sum_wdy_xhat / cols;
  __syncthreads();

  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int index = row * cols + col;
    const float xhat = (x[index] - mean[row]) * rstd[row];
    const float wdy = weight[col] * dy[index];
    dx[index] = rstd[row] *
        (wdy - mean_wdy - xhat * mean_wdy_xhat);
  }

  if (threadIdx.x == 0) {
    while (atomicCAS(locks_and_counts + lock_id, 0, 1) != 0) {
    }
    first_writer = atomicExch(
        locks_and_counts + group_size + lock_id, 1) == 0;
  }
  __syncthreads();
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int source = row * cols + col;
    const int target = lock_id * cols + col;
    const float xhat = (x[source] - mean[row]) * rstd[row];
    const float row_dw = dy[source] * xhat;
    const float row_db = dy[source];
    if (first_writer) {
      grouped_dw[target] = row_dw;
      grouped_db[target] = row_db;
    } else {
      grouped_dw[target] += row_dw;
      grouped_db[target] += row_db;
    }
  }
  // Every writer thread orders its own global stores before thread 0 publishes
  // the unlock.  A fence only in thread 0 would not cover sibling writes.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicExch(locks_and_counts + lock_id, 0);
  }
}

}  // namespace

cudaError_t launch_op005_layer_norm_forward(
    const float* x, const float* weight, const float* bias, float* y,
    float* mean, float* rstd, int rows, int cols, float eps,
    cudaStream_t stream) {
  if (rows < 0 || cols <= 0 || !std::isfinite(eps) || eps < 0.0f)
    return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  if (x == nullptr || weight == nullptr || bias == nullptr || y == nullptr ||
      mean == nullptr || rstd == nullptr)
    return cudaErrorInvalidValue;
  op005_layer_norm_forward_kernel<<<rows, kThreads, 0, stream>>>(
      x, weight, bias, y, mean, rstd, cols, eps);
  return cudaGetLastError();
}

cudaError_t launch_op005_layer_norm_backward(
    const float* dy, const float* x, const float* weight, const float* mean,
    const float* rstd, float* dx, float* partial_dw, float* partial_db,
    float* dw, float* db, int rows, int cols, cudaStream_t stream) {
  if (rows < 0 || cols <= 0) return cudaErrorInvalidValue;
  if (rows == 0) {
    if (dw == nullptr || db == nullptr) return cudaErrorInvalidValue;
    CUDA_CHECK(cudaMemsetAsync(dw, 0, cols * sizeof(float), stream));
    return cudaMemsetAsync(db, 0, cols * sizeof(float), stream);
  }
  if (dy == nullptr || x == nullptr || weight == nullptr || mean == nullptr ||
      rstd == nullptr || dx == nullptr || partial_dw == nullptr ||
      partial_db == nullptr || dw == nullptr || db == nullptr)
    return cudaErrorInvalidValue;
  op005_layer_norm_backward_rows_kernel<<<rows, kThreads, 0, stream>>>(
      dy, x, weight, mean, rstd, dx, partial_dw, partial_db, cols);
  CUDA_CHECK(cudaGetLastError());
  op005_layer_norm_backward_params_kernel<<<cols, kThreads, 0, stream>>>(
      partial_dw, partial_db, dw, db, rows, cols);
  return cudaGetLastError();
}

cudaError_t launch_op005_layer_norm_backward_grouped(
    const float* dy, const float* x, const float* weight, const float* mean,
    const float* rstd, float* dx, float* grouped_dw, float* grouped_db,
    int* locks_and_counts, float* dw, float* db, int rows, int cols,
    int group_size, cudaStream_t stream) {
  if (rows < 0 || cols <= 0 || group_size <= 0)
    return cudaErrorInvalidValue;
  const int active_groups = rows < group_size ? rows : group_size;
  if (rows == 0) {
    if (dw == nullptr || db == nullptr) return cudaErrorInvalidValue;
    CUDA_CHECK(cudaMemsetAsync(dw, 0, cols * sizeof(float), stream));
    return cudaMemsetAsync(db, 0, cols * sizeof(float), stream);
  }
  if (dy == nullptr || x == nullptr || weight == nullptr || mean == nullptr ||
      rstd == nullptr || dx == nullptr || grouped_dw == nullptr ||
      grouped_db == nullptr || locks_and_counts == nullptr || dw == nullptr ||
      db == nullptr)
    return cudaErrorInvalidValue;
  CUDA_CHECK(cudaMemsetAsync(
      locks_and_counts, 0, 2 * group_size * sizeof(int), stream));
  op005_layer_norm_backward_grouped_kernel<<<rows, kThreads, 0, stream>>>(
      dy, x, weight, mean, rstd, dx, grouped_dw, grouped_db,
      locks_and_counts, cols, group_size);
  CUDA_CHECK(cudaGetLastError());
  op005_layer_norm_backward_params_kernel<<<cols, kThreads, 0, stream>>>(
      grouped_dw, grouped_db, dw, db, active_groups, cols);
  return cudaGetLastError();
}

}  // namespace gpu_ops
