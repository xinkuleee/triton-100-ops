"""037: row-wise minimum."""

import torch
import triton
import triton.language as tl

from ._common import _reduction_meta, _require_matrix


@triton.jit
def _op037_row_min_kernel(x_ptr, out_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(
        x_ptr + row * cols + offsets,
        mask=offsets < cols,
        other=float("inf"),
    )
    tl.store(out_ptr + row, tl.min(values, axis=0))


def op037_row_min(x: torch.Tensor) -> torch.Tensor:
    rows, cols = _require_matrix(x)
    out = torch.empty(rows, device=x.device, dtype=x.dtype)
    if rows:
        block, warps = _reduction_meta(cols)
        _op037_row_min_kernel[(rows,)](
            x, out, cols, BLOCK=block, num_warps=warps
        )
    return out


__all__ = ["op037_row_min"]
