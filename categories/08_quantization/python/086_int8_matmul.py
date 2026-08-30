"""086: dequantized INT8 matrix multiplication."""

import torch
import triton
import triton.language as tl

from ._common import (
    MAX_SAFE_INT8_DOT_K,
    _require_cuda_contiguous,
    _require_floating,
    _require_scalar_scale,
)


@triton.jit
def _op086_int8_matmul_kernel(
    a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k_start in range(0, K, BLOCK_K):
        inner = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + row[:, None] * K + inner[None, :],
            mask=(row[:, None] < M) & (inner[None, :] < K), other=0,
        )
        b = tl.load(
            b_ptr + inner[:, None] * N + col[None, :],
            mask=(inner[:, None] < K) & (col[None, :] < N), other=0,
        )
        accumulator += tl.dot(a, b, out_dtype=tl.int32)
    scale = tl.load(a_scale_ptr).to(tl.float32) * tl.load(b_scale_ptr).to(tl.float32)
    tl.store(
        out_ptr + row[:, None] * N + col[None, :],
        accumulator.to(tl.float32) * scale,
        mask=(row[:, None] < M) & (col[None, :] < N),
    )


def op086_int8_matmul(
    a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor
) -> torch.Tensor:
    """Compute dequantized ``A[M,K] @ B[K,N]`` with INT32 accumulation."""
    _require_cuda_contiguous(a, b, a_scale, b_scale)
    _require_floating(a_scale, b_scale)
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("a and b must have dtype int8")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected a[M,K] and b[K,N]")
    _require_scalar_scale(a_scale)
    _require_scalar_scale(b_scale)
    m, k = a.shape
    n = b.shape[1]
    if k < 1:
        raise ValueError("K must be positive")
    if k > MAX_SAFE_INT8_DOT_K:
        raise ValueError(
            f"K must not exceed {MAX_SAFE_INT8_DOT_K}: larger worst-case "
            "[-127,127] dot products can overflow int32 accumulation"
        )
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    if m and n:
        _op086_int8_matmul_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
            a, b, a_scale, b_scale, out, M=m, N=n, K=k,
            BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, num_warps=4, num_stages=3,
        )
    return out
