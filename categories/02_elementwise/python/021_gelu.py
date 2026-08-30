"""021: tanh-approximate GELU."""

import torch
import triton
import triton.language as tl

from ._common import BLOCK_SIZE, grid, require_float_tensor, stable_sigmoid


@triton.jit
def _op021_gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_inner = 2.0 * stable_sigmoid(2.0 * inner) - 1.0
    tl.store(out_ptr + offsets, 0.5 * x * (1.0 + tanh_inner), mask=mask)


def op021_gelu(x: torch.Tensor) -> torch.Tensor:
    """Match ``torch.nn.functional.gelu(..., approximate="tanh")``."""
    require_float_tensor(x, "x")
    out = torch.empty_like(x)
    if x.numel():
        _op021_gelu_kernel[grid(x.numel())](x, out, x.numel(), BLOCK=BLOCK_SIZE)
    return out
