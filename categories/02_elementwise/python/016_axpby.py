"""016: fused ``alpha * x + beta * y``."""

import torch
import triton
import triton.language as tl

from ._common import BLOCK_SIZE, grid, require_pair


@triton.jit
def _op016_axpby_kernel(
    x_ptr, y_ptr, alpha, beta, out_ptr, n_elements, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, alpha * x + beta * y, mask=mask)


def op016_axpby(
    x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float
) -> torch.Tensor:
    require_pair(x, y)
    out = torch.empty_like(x)
    if x.numel():
        _op016_axpby_kernel[grid(x.numel())](
            x, y, alpha, beta, out, x.numel(), BLOCK=BLOCK_SIZE
        )
    return out
