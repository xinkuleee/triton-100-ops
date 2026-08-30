"""050: atomic in-place per-row indexed addition."""

import torch
import triton
import triton.language as tl

from ._common import require_device_tensor, require_same_device


@triton.jit
def _op050_scatter_add_rows_kernel(
    src_ptr, indices_ptr, out_ptr, rows, cols, count, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < rows * count
    row = offset // count
    index = tl.load(indices_ptr + offset, mask=mask, other=0).to(tl.int64)
    value = tl.load(src_ptr + offset, mask=mask, other=0.0)
    tl.atomic_add(
        out_ptr + row * cols + index,
        value,
        mask=mask & (index >= 0) & (index < cols),
    )


def op050_scatter_add_rows_(
    out: torch.Tensor, indices: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    """Atomically add ``src[R,I]`` into selected columns of ``out[R,C]``."""
    require_device_tensor(out, "out")
    require_device_tensor(src, "src")
    require_device_tensor(indices, "indices")
    require_same_device(out, indices, src)
    if out.dtype != torch.float32 or src.dtype != torch.float32:
        raise TypeError("the teaching atomic-add path requires float32 out and src")
    if out.ndim != 2 or indices.ndim != 2 or indices.dtype != torch.int64:
        raise ValueError("expected out[R,C] and int64 indices[R,I]")
    rows, cols = out.shape
    if indices.shape[0] != rows or src.shape != indices.shape:
        raise ValueError("src and indices must be [R,I]")
    count = indices.shape[1]
    total = rows * count
    if total:
        _op050_scatter_add_rows_kernel[(triton.cdiv(total, 256),)](
            src, indices, out, rows, cols, count, BLOCK=256
        )
    return out
