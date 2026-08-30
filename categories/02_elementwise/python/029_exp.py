"""029: elementwise exponential."""

import torch
import triton
import triton.language as tl

from ._common import BLOCK_SIZE, grid, require_float_tensor


@triton.jit
def _op029_exp_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    tl.store(out_ptr + offsets, tl.exp(x), mask=mask)


def op029_exp(x: torch.Tensor) -> torch.Tensor:
    require_float_tensor(x, "x")
    out = torch.empty_like(x)
    if x.numel():
        _op029_exp_kernel[grid(x.numel())](x, out, x.numel(), BLOCK=BLOCK_SIZE)
    return out
