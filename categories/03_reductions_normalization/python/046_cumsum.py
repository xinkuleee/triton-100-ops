"""046: row-wise inclusive prefix sum."""

import torch
import triton
import triton.language as tl

from ._common import _reduction_meta, _require_matrix


@triton.jit
def _op046_cumsum_kernel(x_ptr, out_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < cols
    values = tl.load(
        x_ptr + row * cols + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    inclusive_prefix = tl.cumsum(values, axis=0)
    tl.store(out_ptr + row * cols + offsets, inclusive_prefix, mask=mask)


def op046_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Inclusive prefix sum over the final dimension of x."""
    rows, cols = _require_matrix(x)
    out = torch.empty_like(x)
    if rows:
        block, warps = _reduction_meta(cols)
        _op046_cumsum_kernel[(rows,)](
            x, out, cols, BLOCK=block, num_warps=warps
        )
    return out


__all__ = ["op046_cumsum"]
