"""040: row-wise population variance."""

import torch
import triton
import triton.language as tl

from ._common import _reduction_meta, _require_matrix


@triton.jit
def _op040_variance_kernel(x_ptr, out_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < cols
    values = tl.load(
        x_ptr + row * cols + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    mean = tl.sum(values, axis=0) / cols
    centered = tl.where(mask, values - mean, 0.0)
    tl.store(out_ptr + row, tl.sum(centered * centered, axis=0) / cols)


def op040_variance(x: torch.Tensor) -> torch.Tensor:
    """Return population variance accumulated in FP32."""
    rows, cols = _require_matrix(x)
    out = torch.empty(rows, device=x.device, dtype=torch.float32)
    if rows:
        block, warps = _reduction_meta(cols)
        _op040_variance_kernel[(rows,)](
            x, out, cols, BLOCK=block, num_warps=warps
        )
    return out


__all__ = ["op040_variance"]
