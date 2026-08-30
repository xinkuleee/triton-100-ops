"""052: one-hot encoding."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import require_device_tensor


@triton.jit
def _op052_one_hot_kernel(
    indices_ptr, out_ptr, count, classes, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count * classes
    item = offset // classes
    cls = offset % classes
    index = tl.load(indices_ptr + item, mask=mask, other=-1)
    tl.store(out_ptr + offset, tl.where(cls == index, 1.0, 0.0), mask=mask)


def op052_one_hot(indices: torch.Tensor, classes: int) -> torch.Tensor:
    """Create FP32 one-hot rows; out-of-range indices produce all-zero rows."""
    require_device_tensor(indices, "indices")
    if indices.ndim != 1 or indices.dtype != torch.int64:
        raise ValueError("indices must be a contiguous int64 vector")
    if classes <= 0:
        raise ValueError("classes must be positive")
    count = indices.numel()
    out = torch.empty((count, classes), device=indices.device, dtype=torch.float32)
    total = count * classes
    if total:
        _op052_one_hot_kernel[(triton.cdiv(total, 256),)](
            indices, out, count, classes, BLOCK=256
        )
    return out
