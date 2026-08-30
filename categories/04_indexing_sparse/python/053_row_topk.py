"""053: row-wise Top-K selection."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import require_float_tensor, row_launch_meta


@triton.jit
def _op053_row_topk_kernel(
    x_ptr, values_ptr, indices_ptr, cols,
    K: tl.constexpr, BLOCK_WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_WIDTH)
    working = tl.load(
        x_ptr + row * cols + col, mask=col < cols, other=-float("inf")
    ).to(tl.float32)
    for rank in tl.static_range(0, K):
        index = tl.argmax(working, axis=0, tie_break_left=True)
        value = tl.max(working, axis=0)
        tl.store(values_ptr + row * K + rank, value)
        tl.store(indices_ptr + row * K + rank, index)
        working = tl.where(col == index, -float("inf"), working)


def op053_row_topk(
    x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return descending row top-k; this readable baseline costs O(k*C)."""
    require_float_tensor(x, "x")
    if x.ndim != 2:
        raise ValueError("x must have shape [rows, cols]")
    rows, cols = x.shape
    # This safety check synchronizes; a trusted-finite production path can skip it.
    if not torch.isfinite(x).all().item():
        raise ValueError("the teaching Top-K path requires finite inputs")
    block, warps = row_launch_meta(cols)
    if k < 1 or k > min(cols, 64):
        raise ValueError("k must be in [1, min(cols, 64)]")
    values = torch.empty((rows, k), device=x.device, dtype=x.dtype)
    indices = torch.empty((rows, k), device=x.device, dtype=torch.int64)
    if rows:
        _op053_row_topk_kernel[(rows,)](
            x, values, indices, cols, K=k, BLOCK_WIDTH=block, num_warps=warps
        )
    return values, indices
