"""019: overflow-safe sigmoid."""

import torch
import triton
import triton.language as tl

from ._common import BLOCK_SIZE, grid, require_float_tensor, stable_sigmoid


@triton.jit
def _op019_sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    tl.store(out_ptr + offsets, stable_sigmoid(x), mask=mask)


def op019_sigmoid(x: torch.Tensor) -> torch.Tensor:
    require_float_tensor(x, "x")
    out = torch.empty_like(x)
    if x.numel():
        _op019_sigmoid_kernel[grid(x.numel())](x, out, x.numel(), BLOCK=BLOCK_SIZE)
    return out
