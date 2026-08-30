"""032: choose elements from two tensors using a boolean mask."""

import torch
import triton
import triton.language as tl

from ._common import BLOCK_SIZE, grid, require_pair


@triton.jit
def _op032_where_kernel(
    condition_ptr, x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    condition = tl.load(condition_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.where(condition != 0, x, y), mask=mask)


def op032_where(
    condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    require_pair(x, y)
    if condition.dtype not in (torch.bool, torch.uint8):
        raise TypeError("condition must have dtype torch.bool or torch.uint8")
    if (
        not condition.is_cuda
        or not condition.is_contiguous()
        or condition.shape != x.shape
        or condition.device != x.device
    ):
        raise ValueError("condition must be contiguous and match x shape/device")
    out = torch.empty_like(x)
    if x.numel():
        _op032_where_kernel[grid(x.numel())](
            condition, x, y, out, x.numel(), BLOCK=BLOCK_SIZE
        )
    return out
