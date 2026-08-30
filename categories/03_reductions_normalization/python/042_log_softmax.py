"""042: row-wise log-softmax."""

import torch
import triton
import triton.language as tl

from ._common import _reduction_meta, _require_matrix


@triton.jit
def _op042_log_softmax_kernel(x_ptr, out_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < cols
    values = tl.load(
        x_ptr + row * cols + offsets, mask=mask, other=-float("inf")
    ).to(tl.float32)
    maximum = tl.max(values, axis=0)
    log_denominator = maximum + tl.log(
        tl.sum(tl.exp(values - maximum), axis=0)
    )
    tl.store(
        out_ptr + row * cols + offsets, values - log_denominator, mask=mask
    )


def op042_log_softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = _require_matrix(x)
    out = torch.empty_like(x)
    if rows:
        block, warps = _reduction_meta(cols)
        _op042_log_softmax_kernel[(rows,)](
            x, out, cols, BLOCK=block, num_warps=warps
        )
    return out


__all__ = ["op042_log_softmax"]
