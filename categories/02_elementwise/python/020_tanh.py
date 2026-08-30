"""020: hyperbolic tangent via stable sigmoid."""

import torch
import triton
import triton.language as tl

from ._common import BLOCK_SIZE, grid, require_float_tensor, stable_sigmoid


@triton.jit
def _op020_tanh_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    tl.store(out_ptr + offsets, 2.0 * stable_sigmoid(2.0 * x) - 1.0, mask=mask)


def op020_tanh(x: torch.Tensor) -> torch.Tensor:
    require_float_tensor(x, "x")
    out = torch.empty_like(x)
    if x.numel():
        _op020_tanh_kernel[grid(x.numel())](x, out, x.numel(), BLOCK=BLOCK_SIZE)
    return out
