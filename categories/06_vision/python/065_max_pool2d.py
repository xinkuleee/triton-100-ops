"""065: NCHW max pooling."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import (
    kernel_pair,
    output_extent,
    reduction_meta,
    require_stride_padding,
    require_visual_tensors,
)


@triton.jit
def _op065_max_pool2d_kernel(
    x_ptr,
    out_ptr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    OUT_HEIGHT: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    KERNEL_HEIGHT: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    BLOCK_KERNEL: tl.constexpr,
):
    output_index = tl.program_id(0)
    output_x = output_index % OUT_WIDTH
    output_y = (output_index // OUT_WIDTH) % OUT_HEIGHT
    batch_channel = output_index // (OUT_HEIGHT * OUT_WIDTH)
    kernel_offset = tl.arange(0, BLOCK_KERNEL)
    kernel_y = kernel_offset // KERNEL_WIDTH
    kernel_x = kernel_offset % KERNEL_WIDTH
    input_y = output_y * STRIDE + kernel_y - PADDING
    input_x = output_x * STRIDE + kernel_x - PADDING
    valid = (
        (kernel_offset < KERNEL_HEIGHT * KERNEL_WIDTH)
        & (input_y >= 0)
        & (input_y < HEIGHT)
        & (input_x >= 0)
        & (input_x < WIDTH)
    )
    values = tl.load(
        x_ptr + (batch_channel * HEIGHT + input_y) * WIDTH + input_x,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    tl.store(out_ptr + output_index, tl.max(values, axis=0))


def op065_max_pool2d(
    x: torch.Tensor,
    kernel: int | tuple[int, int] = (2, 2),
    stride: int = 2,
    padding: int = 0,
) -> torch.Tensor:
    """Max-pool a contiguous NCHW tensor; padded samples equal ``-inf``."""
    require_visual_tensors(x)
    require_stride_padding(stride, padding)
    if x.ndim != 4:
        raise ValueError("expected x[N,C,H,W]")
    batch, channels, height, width = x.shape
    if channels <= 0:
        raise ValueError("channels must be positive")
    kernel_height, kernel_width = kernel_pair(kernel)
    kernel_elements = kernel_height * kernel_width
    out_height = output_extent(height, kernel_height, stride, padding)
    out_width = output_extent(width, kernel_width, stride, padding)
    out = torch.empty(
        (batch, channels, out_height, out_width),
        device=x.device,
        dtype=torch.float32,
    )
    if out.numel():
        block_kernel, warps = reduction_meta(kernel_elements, "kernel area")
        _op065_max_pool2d_kernel[(out.numel(),)](
            x,
            out,
            HEIGHT=height,
            WIDTH=width,
            OUT_HEIGHT=out_height,
            OUT_WIDTH=out_width,
            KERNEL_HEIGHT=kernel_height,
            KERNEL_WIDTH=kernel_width,
            STRIDE=stride,
            PADDING=padding,
            BLOCK_KERNEL=block_kernel,
            num_warps=warps,
        )
    return out
