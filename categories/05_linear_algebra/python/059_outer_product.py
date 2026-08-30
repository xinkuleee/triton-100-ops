"""059: outer product of two vectors."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import require_tensors


@triton.jit
def _op059_outer_product_kernel(
    x_ptr, y_ptr, out_ptr, rows, cols, BM: tl.constexpr, BN: tl.constexpr
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    x = tl.load(x_ptr + row, mask=row < rows, other=0.0)
    y = tl.load(y_ptr + col, mask=col < cols, other=0.0)
    tl.store(
        out_ptr + row[:, None] * cols + col[None, :],
        x[:, None].to(tl.float32) * y[None, :],
        mask=(row[:, None] < rows) & (col[None, :] < cols),
    )


def op059_outer_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the Cartesian product ``x[:, None] * y[None, :]``."""
    require_tensors(x, y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("expected two vectors")
    rows, cols = x.numel(), y.numel()
    out = torch.empty((rows, cols), device=x.device, dtype=torch.float32)
    if rows and cols:
        grid = (triton.cdiv(rows, 32), triton.cdiv(cols, 32))
        _op059_outer_product_kernel[grid](
            x, y, out, rows, cols, BM=32, BN=32, num_warps=4
        )
    return out
