"""041: row-wise argmax with leftmost tie breaking."""

import torch
import triton
import triton.language as tl

from ._common import _reduction_meta, _require_matrix


@triton.jit
def _op041_argmax_kernel(x_ptr, out_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(
        x_ptr + row * cols + offsets,
        mask=offsets < cols,
        other=-float("inf"),
    )
    index = tl.argmax(values, axis=0, tie_break_left=True)
    tl.store(out_ptr + row, index)


def op041_argmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = _require_matrix(x)
    out = torch.empty(rows, device=x.device, dtype=torch.int64)
    if rows:
        block, warps = _reduction_meta(cols)
        _op041_argmax_kernel[(rows,)](
            x, out, cols, BLOCK=block, num_warps=warps
        )
    return out


__all__ = ["op041_argmax"]
