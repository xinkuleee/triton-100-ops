"""057: batched dense matrix multiplication."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import MAX_DOT_K, require_tensors


@triton.jit
def _op057_batched_matmul_kernel(
    a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.program_id(1) * BM + tl.arange(0, BM)
    cols = tl.program_id(2) * BN + tl.arange(0, BN)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        inner = k0 + tl.arange(0, BK)
        a = tl.load(
            a_ptr + batch * M * K + rows[:, None] * K + inner[None, :],
            mask=(rows[:, None] < M) & (inner[None, :] < K), other=0.0,
        )
        b = tl.load(
            b_ptr + batch * K * N + inner[:, None] * N + cols[None, :],
            mask=(inner[:, None] < K) & (cols[None, :] < N), other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)
    tl.store(
        out_ptr + batch * M * N + rows[:, None] * N + cols[None, :],
        accumulator, mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def op057_batched_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute ``out[b] = a[b] @ b[b]`` for dense row-major batches."""
    require_tensors(a, b)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise ValueError("expected a[B,M,K] and b[B,K,N]")
    batch, m, k = a.shape
    n = b.shape[2]
    if k > MAX_DOT_K:
        raise ValueError(f"K must not exceed {MAX_DOT_K}")
    out = torch.empty((batch, m, n), device=a.device, dtype=torch.float32)
    if batch and m and n:
        grid = (batch, triton.cdiv(m, 32), triton.cdiv(n, 32))
        _op057_batched_matmul_kernel[grid](
            a, b, out, M=m, N=n, K=k, BM=32, BN=32, BK=32,
            num_warps=4, num_stages=3,
        )
    return out
