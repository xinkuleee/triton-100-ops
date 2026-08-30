"""033: add a column bias to every row of a matrix."""

import torch
import triton
import triton.language as tl

from ._common import BLOCK_SIZE, grid, require_float_tensor


@triton.jit
def _op033_row_bias_add_kernel(
    x_ptr, bias_ptr, out_ptr, n_elements, cols, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets % cols, mask=mask)
    tl.store(out_ptr + offsets, x + bias, mask=mask)


def op033_row_bias_add(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    require_float_tensor(x, "x")
    require_float_tensor(bias, "bias")
    if x.ndim != 2 or bias.ndim != 1 or bias.shape[0] != x.shape[1]:
        raise ValueError("expected x[rows, cols] and bias[cols]")
    if x.device != bias.device or x.dtype != bias.dtype:
        raise ValueError("x and bias must have the same device and dtype")
    out = torch.empty_like(x)
    if x.numel():
        _op033_row_bias_add_kernel[grid(x.numel())](
            x, bias, out, x.numel(), x.shape[1], BLOCK=BLOCK_SIZE
        )
    return out
