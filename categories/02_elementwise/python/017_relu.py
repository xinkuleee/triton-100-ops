"""017: ReLU activation."""

import torch
import triton
import triton.language as tl

from ._common import BLOCK_SIZE, grid, require_float_tensor


@triton.jit
def _op017_relu_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.where(x < 0.0, 0.0, x)
    tl.store(out_ptr + offsets, result, mask=mask)


def op017_relu(x: torch.Tensor) -> torch.Tensor:
    require_float_tensor(x, "x")
    out = torch.empty_like(x)
    if x.numel():
        _op017_relu_kernel[grid(x.numel())](x, out, x.numel(), BLOCK=BLOCK_SIZE)
    return out
