"""055: COO scatter-add."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import require_device_tensor, require_same_device


@triton.jit
def _op055_coo_scatter_add_kernel(
    indices_ptr, values_ptr, out_ptr, nnz, size, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < nnz
    index = tl.load(indices_ptr + offset, mask=mask, other=0)
    value = tl.load(values_ptr + offset, mask=mask, other=0.0)
    tl.atomic_add(out_ptr + index, value, mask=mask & (index >= 0) & (index < size))


def op055_coo_scatter_add(
    indices: torch.Tensor, values: torch.Tensor, size: int
) -> torch.Tensor:
    """Accumulate 1-D COO entries; invalid destinations are ignored."""
    require_device_tensor(indices, "indices")
    require_device_tensor(values, "values")
    require_same_device(values, indices)
    if indices.dtype != torch.int32 or indices.ndim != 1:
        raise ValueError("indices must be a contiguous int32 vector")
    if values.dtype != torch.float32 or values.ndim != 1:
        raise ValueError("values must be a contiguous float32 vector")
    if indices.shape != values.shape or size < 0:
        raise ValueError("indices/values shapes must match and size must be nonnegative")
    out = torch.zeros(size, device=values.device, dtype=torch.float32)
    if values.numel() and size:
        _op055_coo_scatter_add_kernel[(triton.cdiv(values.numel(), 256),)](
            indices, values, out, values.numel(), size, BLOCK=256
        )
    return out
