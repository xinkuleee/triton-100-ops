"""038: row-wise L1 norm."""

import torch
import triton
import triton.language as tl

from ._common import _reduction_meta, _require_matrix


@triton.jit
def _op038_l1_norm_kernel(x_ptr, out_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(
        x_ptr + row * cols + offsets, mask=offsets < cols, other=0.0
    ).to(tl.float32)
    tl.store(out_ptr + row, tl.sum(tl.abs(values), axis=0))


def op038_l1_norm(x: torch.Tensor) -> torch.Tensor:
    rows, cols = _require_matrix(x)
    out = torch.empty(rows, device=x.device, dtype=torch.float32)
    if rows:
        block, warps = _reduction_meta(cols)
        _op038_l1_norm_kernel[(rows,)](
            x, out, cols, BLOCK=block, num_warps=warps
        )
    return out


__all__ = ["op038_l1_norm"]
