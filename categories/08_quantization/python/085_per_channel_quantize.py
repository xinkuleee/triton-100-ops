"""085: symmetric INT8 quantization with one scale per column."""

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
def _op085_per_channel_quantize_kernel(
    x_ptr, q_ptr, scales_ptr, rows, cols, BLOCK: tl.constexpr
):
    col = tl.program_id(0)
    row = tl.arange(0, BLOCK)
    mask = row < rows
    value = tl.load(x_ptr + row * cols + col, mask=mask, other=0.0).to(tl.float32)
    absolute_maximum = tl.max(tl.abs(value), axis=0)
    scale = tl.maximum(absolute_maximum / 127.0, 1.0e-12)
    tl.store(
        q_ptr + row * cols + col, _op082_round_and_clamp_int8(value / scale), mask=mask
    )
    tl.store(scales_ptr + col, scale)


def op085_per_channel_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a row-major matrix with one scale per column/channel."""
    _require_cuda_contiguous(x)
    _require_floating(x)
    if x.ndim != 2:
        raise ValueError("x must have shape [rows, channels]")
    rows, cols = x.shape
    block, warps = _reduction_meta(rows)
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(cols, device=x.device, dtype=torch.float32)
    if cols:
        _op085_per_channel_quantize_kernel[(cols,)](
            x, q, scales, rows, cols, BLOCK=block, num_warps=warps
        )
    return q, scales
