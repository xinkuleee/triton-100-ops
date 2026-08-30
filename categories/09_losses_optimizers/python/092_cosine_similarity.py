"""092: row-wise cosine similarity."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ._common import require_cuda_contiguous, require_floating, row_meta


@triton.jit
def _op092_cosine_similarity_kernel(
    a_ptr, b_ptr, out_ptr, cols, eps, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < cols
    a = tl.load(a_ptr + row * cols + col, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * cols + col, mask=mask, other=0.0).to(tl.float32)
    dot = tl.sum(a * b, axis=0)
    squared_a = tl.sum(a * a, axis=0)
    squared_b = tl.sum(b * b, axis=0)
    denominator = tl.maximum(tl.sqrt(squared_a), eps) * tl.maximum(
        tl.sqrt(squared_b), eps
    )
    tl.store(out_ptr + row, dot / denominator)


def op092_cosine_similarity(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Return row-wise cosine similarity for matching ``[R,C]`` tensors."""
    require_cuda_contiguous(a, b)
    require_floating(a, b)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("a and b must have matching shape [rows, cols]")
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    rows, cols = a.shape
    block, warps = row_meta(cols)
    out = torch.empty(rows, device=a.device, dtype=torch.float32)
    if rows:
        _op092_cosine_similarity_kernel[(rows,)](
            a, b, out, cols, eps, BLOCK=block, num_warps=warps
        )
    return out
