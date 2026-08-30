"""012: elementwise subtraction."""

import torch
import triton
import triton.language as tl

from ._common import BLOCK_SIZE, grid, require_pair


@triton.jit
def _op012_sub_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a - b, mask=mask)


def op012_sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    require_pair(x, y)
    out = torch.empty_like(x)
    if x.numel():
        _op012_sub_kernel[grid(x.numel())](x, y, out, x.numel(), BLOCK=BLOCK_SIZE)
    return out
