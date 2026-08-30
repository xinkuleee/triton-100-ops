"""026: hard-swish activation."""

import torch
import triton
import triton.language as tl

from ._common import BLOCK_SIZE, grid, require_float_tensor


@triton.jit
def _op026_hard_swish_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    value = x / 6.0 + 0.5
    gate = tl.where(value < 0.0, 0.0, tl.where(value > 1.0, 1.0, value))
    tl.store(out_ptr + offsets, x * gate, mask=mask)


def op026_hard_swish(x: torch.Tensor) -> torch.Tensor:
    require_float_tensor(x, "x")
    out = torch.empty_like(x)
    if x.numel():
        _op026_hard_swish_kernel[grid(x.numel())](
            x, out, x.numel(), BLOCK=BLOCK_SIZE
        )
    return out
