"""084: symmetric INT8 quantization with one scale per row."""

import torch
import triton
import triton.language as tl

from ._common import (
    _reduction_meta,
    _require_cuda_contiguous,
    _require_floating,
    _op082_round_and_clamp_int8,
)


@triton.jit
def _op084_per_row_quantize_kernel(
    x_ptr, q_ptr, scales_ptr, cols, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < cols
    value = tl.load(x_ptr + row * cols + col, mask=mask, other=0.0).to(tl.float32)
    absolute_maximum = tl.max(tl.abs(value), axis=0)
    scale = tl.maximum(absolute_maximum / 127.0, 1.0e-12)
    tl.store(
        q_ptr + row * cols + col, _op082_round_and_clamp_int8(value / scale), mask=mask
    )
    tl.store(scales_ptr + row, scale)


def op084_per_row_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one scale and symmetric INT8 values for every row of ``x``."""
    _require_cuda_contiguous(x)
    _require_floating(x)
    if x.ndim != 2:
        raise ValueError("x must have shape [rows, cols]")
    rows, cols = x.shape
    block, warps = _reduction_meta(cols)
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(rows, device=x.device, dtype=torch.float32)
    if rows:
        _op084_per_row_quantize_kernel[(rows,)](
            x, q, scales, cols, BLOCK=block, num_warps=warps
        )
    return q, scales
