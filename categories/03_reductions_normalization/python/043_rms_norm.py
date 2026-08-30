"""043: RMS normalization."""

import math

import torch
import triton
import triton.language as tl

from ._common import (
    _reduction_meta,
    _require_cuda_contiguous,
    _require_float,
    _require_matrix,
    _require_same_device,
    _require_same_dtype,
)


@triton.jit
def _op043_rms_norm_kernel(
    x_ptr, gamma_ptr, out_ptr, cols, eps, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < cols
    values = tl.load(
        x_ptr + row * cols + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / cols
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = values * tl.rsqrt(mean_square + eps) * gamma
    tl.store(out_ptr + row * cols + offsets, result, mask=mask)


def op043_rms_norm(
    x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    rows, cols = _require_matrix(x)
    _require_cuda_contiguous(gamma, "gamma")
    _require_float(gamma, "gamma")
    _require_same_device(x, gamma, "gamma")
    _require_same_dtype(x, gamma, "gamma")
    if gamma.shape != (cols,):
        raise ValueError("gamma must have shape [cols]")
    if eps < 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and nonnegative")
    out = torch.empty_like(x)
    if rows:
        block, warps = _reduction_meta(cols)
        _op043_rms_norm_kernel[(rows,)](
            x, gamma, out, cols, eps, BLOCK=block, num_warps=warps
        )
    return out


__all__ = ["op043_rms_norm"]
