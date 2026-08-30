"""075: fused residual addition and RMSNorm."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ._common import _require_cuda_contiguous, _require_floating, _row_meta


@triton.jit
def _op075_residual_rms_norm_kernel(
    x_ptr, residual_ptr, gamma_ptr, out_ptr, cols, eps, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < cols
    z = tl.load(x_ptr + row * cols + col, mask=mask, other=0.0).to(tl.float32)
    z += tl.load(residual_ptr + row * cols + col, mask=mask, other=0.0).to(tl.float32)
    square_mean = tl.sum(tl.where(mask, z * z, 0.0), axis=0) / cols
    gamma = tl.load(gamma_ptr + col, mask=mask, other=0.0)
    tl.store(out_ptr + row * cols + col, z * tl.rsqrt(square_mean + eps) * gamma, mask=mask)


def op075_residual_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    _require_cuda_contiguous(x, residual, gamma)
    _require_floating(x, residual, gamma)
    if x.ndim != 2 or residual.shape != x.shape:
        raise ValueError("x and residual must have matching shape [rows, cols]")
    rows, cols = x.shape
    if gamma.shape != (cols,):
        raise ValueError("gamma must have shape [cols]")
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    block, warps = _row_meta(cols)
    out = torch.empty_like(x)
    if rows:
        _op075_residual_rms_norm_kernel[(rows,)](
            x, residual, gamma, out, cols, eps, BLOCK=block, num_warps=warps
        )
    return out
