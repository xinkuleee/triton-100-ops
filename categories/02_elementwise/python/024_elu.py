"""024: ELU activation."""

import torch
import triton
import triton.language as tl

from ._common import BLOCK_SIZE, grid, require_float_tensor


@triton.jit
def _op024_elu_kernel(x_ptr, alpha, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    result = tl.where(x > 0.0, x, alpha * (tl.exp(x) - 1.0))
    tl.store(out_ptr + offsets, result, mask=mask)


def op024_elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    require_float_tensor(x, "x")
    out = torch.empty_like(x)
    if x.numel():
        _op024_elu_kernel[grid(x.numel())](
            x, alpha, out, x.numel(), BLOCK=BLOCK_SIZE
        )
    return out
