"""015: add one scalar to every element."""

import torch
import triton
import triton.language as tl

from ._common import BLOCK_SIZE, grid, require_float_tensor


@triton.jit
def _op015_scalar_add_kernel(x_ptr, scalar, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, values + scalar, mask=mask)


def op015_scalar_add(x: torch.Tensor, scalar: float) -> torch.Tensor:
    require_float_tensor(x, "x")
    out = torch.empty_like(x)
    if x.numel():
        _op015_scalar_add_kernel[grid(x.numel())](
            x, scalar, out, x.numel(), BLOCK=BLOCK_SIZE
        )
    return out
