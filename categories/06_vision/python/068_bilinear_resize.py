"""068: half-pixel bilinear NCHW resize."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import RESIZE_BLOCK, require_visual_tensors


@triton.jit
def _op068_bilinear_resize_kernel(
    x_ptr,
    out_ptr,
    total,
    CHANNELS: tl.constexpr,
    INPUT_HEIGHT: tl.constexpr,
    INPUT_WIDTH: tl.constexpr,
    OUTPUT_HEIGHT: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    output_index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    output_mask = output_index < total
    output_x = output_index % OUTPUT_WIDTH
    output_y = (output_index // OUTPUT_WIDTH) % OUTPUT_HEIGHT
    channel = (output_index // (OUTPUT_HEIGHT * OUTPUT_WIDTH)) % CHANNELS
    batch = output_index // (CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH)

    source_y = (output_y + 0.5) * INPUT_HEIGHT / OUTPUT_HEIGHT - 0.5
    source_x = (output_x + 0.5) * INPUT_WIDTH / OUTPUT_WIDTH - 0.5
    source_y = tl.maximum(source_y, 0.0)
    source_x = tl.maximum(source_x, 0.0)
    y0 = source_y.to(tl.int32)
    x0 = source_x.to(tl.int32)
    y1 = tl.minimum(y0 + 1, INPUT_HEIGHT - 1)
    x1 = tl.minimum(x0 + 1, INPUT_WIDTH - 1)
    y_weight = source_y - y0
    x_weight = source_x - x0
    image_base = (batch * CHANNELS + channel) * INPUT_HEIGHT * INPUT_WIDTH

    top_left = tl.load(
        x_ptr + image_base + y0 * INPUT_WIDTH + x0,
        mask=output_mask,
        other=0.0,
    ).to(tl.float32)
    top_right = tl.load(
        x_ptr + image_base + y0 * INPUT_WIDTH + x1,
        mask=output_mask,
        other=0.0,
    ).to(tl.float32)
    bottom_left = tl.load(
        x_ptr + image_base + y1 * INPUT_WIDTH + x0,
        mask=output_mask,
        other=0.0,
    ).to(tl.float32)
    bottom_right = tl.load(
        x_ptr + image_base + y1 * INPUT_WIDTH + x1,
        mask=output_mask,
        other=0.0,
    ).to(tl.float32)
    top = (1.0 - x_weight) * top_left + x_weight * top_right
    bottom = (1.0 - x_weight) * bottom_left + x_weight * bottom_right
    value = (1.0 - y_weight) * top + y_weight * bottom
    tl.store(out_ptr + output_index, value, mask=output_mask)


def op068_bilinear_resize(
    x: torch.Tensor, out_height: int, out_width: int
) -> torch.Tensor:
    """Bilinear NCHW resize with half-pixel, ``align_corners=False`` coordinates."""
    require_visual_tensors(x)
    if x.ndim != 4:
        raise ValueError("expected x[N,C,H,W]")
    if (
        not isinstance(out_height, int)
        or isinstance(out_height, bool)
        or not isinstance(out_width, int)
        or isinstance(out_width, bool)
        or out_height <= 0
        or out_width <= 0
    ):
        raise ValueError("output height and width must be positive integers")
    batch, channels, input_height, input_width = x.shape
    if channels <= 0 or input_height <= 0 or input_width <= 0:
        raise ValueError("input channels, height, and width must be positive")
    out = torch.empty(
        (batch, channels, out_height, out_width), device=x.device, dtype=x.dtype
    )
    if out.numel():
        _op068_bilinear_resize_kernel[(triton.cdiv(out.numel(), RESIZE_BLOCK),)](
            x,
            out,
            out.numel(),
            CHANNELS=channels,
            INPUT_HEIGHT=input_height,
            INPUT_WIDTH=input_width,
            OUTPUT_HEIGHT=out_height,
            OUTPUT_WIDTH=out_width,
            BLOCK=RESIZE_BLOCK,
            num_warps=4,
        )
    return out
