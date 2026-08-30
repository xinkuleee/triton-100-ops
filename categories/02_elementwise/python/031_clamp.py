"""031: clamp values to a closed interval."""

import torch
import triton
import triton.language as tl

from ._common import BLOCK_SIZE, grid, require_float_tensor


@triton.jit
def _op031_clamp_kernel(
    x_ptr, low, high, out_ptr, n_elements, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.where(x < low, low, tl.where(x > high, high, x))
    tl.store(out_ptr + offsets, result, mask=mask)


def op031_clamp(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    if low > high:
        raise ValueError("low must not exceed high")
    require_float_tensor(x, "x")
    out = torch.empty_like(x)
    if x.numel():
        _op031_clamp_kernel[grid(x.numel())](
            x, low, high, out, x.numel(), BLOCK=BLOCK_SIZE
        )
    return out
