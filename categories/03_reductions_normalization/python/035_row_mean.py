"""035: row-wise mean."""

import torch
import triton
import triton.language as tl

from ._common import _reduction_meta, _require_matrix


@triton.jit
def _op035_row_mean_kernel(x_ptr, out_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(
        x_ptr + row * cols + offsets, mask=offsets < cols, other=0.0
    ).to(tl.float32)
    tl.store(out_ptr + row, tl.sum(values, axis=0) / cols)


def op035_row_mean(x: torch.Tensor) -> torch.Tensor:
    rows, cols = _require_matrix(x)
    out = torch.empty(rows, device=x.device, dtype=torch.float32)
    if rows:
        block, warps = _reduction_meta(cols)
        _op035_row_mean_kernel[(rows,)](
            x, out, cols, BLOCK=block, num_warps=warps
        )
    return out


__all__ = ["op035_row_mean"]
