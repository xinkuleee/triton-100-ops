"""049: in-place per-row indexed scatter."""

import torch
import triton
import triton.language as tl

from ._common import require_device_tensor, require_float_tensor, require_same_device


@triton.jit
def _op049_scatter_rows_kernel(
    src_ptr, indices_ptr, out_ptr, rows, cols, count, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < rows * count
    row = offset // count
    index = tl.load(indices_ptr + offset, mask=mask, other=0).to(tl.int64)
    value = tl.load(src_ptr + offset, mask=mask)
    tl.store(
        out_ptr + row * cols + index,
        value,
        mask=mask & (index >= 0) & (index < cols),
    )


def op049_scatter_rows_(
    out: torch.Tensor, indices: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    """In-place row scatter; duplicate destinations have an undefined winner."""
    require_float_tensor(out, "out")
    require_float_tensor(src, "src")
    require_device_tensor(indices, "indices")
    require_same_device(out, indices, src)
    if out.ndim != 2 or indices.ndim != 2 or indices.dtype != torch.int64:
        raise ValueError("expected out[R,C] and int64 indices[R,I]")
    rows, cols = out.shape
    if indices.shape[0] != rows or src.shape != indices.shape:
        raise ValueError("src and indices must be [R,I]")
    if src.dtype != out.dtype:
        raise ValueError("src and out must have the same dtype")
    count = indices.shape[1]
    total = rows * count
    if total:
        _op049_scatter_rows_kernel[(triton.cdiv(total, 256),)](
            src, indices, out, rows, cols, count, BLOCK=256
        )
    return out
