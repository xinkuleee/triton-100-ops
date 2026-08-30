"""048: per-row indexed gather with zero for invalid column indices."""

import torch
import triton
import triton.language as tl

from ._common import require_device_tensor, require_float_tensor, require_same_device


@triton.jit
def _op048_gather_rows_kernel(
    x_ptr, indices_ptr, out_ptr, rows, cols, count, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < rows * count
    row = offset // count
    index = tl.load(indices_ptr + offset, mask=mask, other=0).to(tl.int64)
    valid = mask & (index >= 0) & (index < cols)
    value = tl.load(x_ptr + row * cols + index, mask=valid, other=0.0)
    tl.store(out_ptr + offset, value, mask=mask)


def op048_gather_rows(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """For each row gather columns named by ``indices[row]``."""
    require_float_tensor(x, "x")
    require_device_tensor(indices, "indices")
    require_same_device(x, indices)
    if x.ndim != 2 or indices.ndim != 2 or indices.dtype != torch.int64:
        raise ValueError("expected x[R,C] and int64 indices[R,I]")
    rows, cols = x.shape
    if indices.shape[0] != rows or cols < 1:
        raise ValueError(
            "indices first dimension must equal rows and C must be positive"
        )
    count = indices.shape[1]
    out = torch.empty((rows, count), device=x.device, dtype=x.dtype)
    total = rows * count
    if total:
        _op048_gather_rows_kernel[(triton.cdiv(total, 256),)](
            x, indices, out, rows, cols, count, BLOCK=256
        )
    return out
