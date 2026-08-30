"""025: hard sigmoid activation."""

import torch
import triton
import triton.language as tl

from ._common import BLOCK_SIZE, grid, require_float_tensor


@triton.jit
def _op025_hard_sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    value = x / 6.0 + 0.5
    result = tl.where(value < 0.0, 0.0, tl.where(value > 1.0, 1.0, value))
    tl.store(out_ptr + offsets, result, mask=mask)


def op025_hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    require_float_tensor(x, "x")
    out = torch.empty_like(x)
    if x.numel():
        _op025_hard_sigmoid_kernel[grid(x.numel())](
            x, out, x.numel(), BLOCK=BLOCK_SIZE
        )
    return out
