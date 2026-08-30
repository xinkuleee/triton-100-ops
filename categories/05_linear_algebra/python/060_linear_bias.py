"""060: linear transform with a fused bias add."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import MAX_DOT_K, require_tensors


@triton.jit
def _op060_linear_bias_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        inner = k0 + tl.arange(0, BK)
        x = tl.load(
            x_ptr + rows[:, None] * K + inner[None, :],
            mask=(rows[:, None] < M) & (inner[None, :] < K), other=0.0,
        )
        # Weight is [N,K], so this pointer tile materializes weight.T.
        weight = tl.load(
            weight_ptr + cols[None, :] * K + inner[:, None],
            mask=(cols[None, :] < N) & (inner[:, None] < K), other=0.0,
        )
        accumulator = tl.dot(x, weight, accumulator)
    bias = tl.load(bias_ptr + cols, mask=cols < N, other=0.0)
    tl.store(
        out_ptr + rows[:, None] * N + cols[None, :],
        accumulator + bias[None, :],
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def op060_linear_bias(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Compute ``x @ weight.T + bias`` while fusing the bias write."""
    require_tensors(x, weight, bias)
    if (x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1
            or x.shape[1] != weight.shape[1] or weight.shape[0] != bias.numel()):
        raise ValueError("expected x[M,K], weight[N,K], and bias[N]")
    m, k = x.shape
    n = weight.shape[0]
    if k > MAX_DOT_K:
        raise ValueError(f"K must not exceed {MAX_DOT_K}")
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    if m and n:
        grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))
        _op060_linear_bias_kernel[grid](
            x, weight, bias, out, M=m, N=n, K=k, BM=32, BN=32, BK=32,
            num_warps=4, num_stages=3,
        )
    return out
