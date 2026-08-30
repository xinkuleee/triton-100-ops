"""062: direct dense NCHW two-dimensional cross-correlation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import (
    MAX_DIRECT_KERNEL_ELEMENTS,
    output_extent,
    reduction_meta,
    require_stride_padding,
    require_visual_tensors,
)


@triton.jit
def _op062_conv2d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    OUT_CHANNELS: tl.constexpr,
    KERNEL_HEIGHT: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    OUT_HEIGHT: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    output_index = tl.program_id(0)
    output_x = output_index % OUT_WIDTH
    output_y = (output_index // OUT_WIDTH) % OUT_HEIGHT
    output_channel = (output_index // (OUT_WIDTH * OUT_HEIGHT)) % OUT_CHANNELS
    batch = output_index // (OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH)

    channel = tl.arange(0, BLOCK_CHANNELS)
    channel_mask = channel < CHANNELS
    accumulator = tl.zeros((BLOCK_CHANNELS,), tl.float32)
    for kernel_y in range(KERNEL_HEIGHT):
        for kernel_x in range(KERNEL_WIDTH):
            input_y = output_y * STRIDE + kernel_y - PADDING
            input_x = output_x * STRIDE + kernel_x - PADDING
            input_mask = (
                channel_mask
                & (input_y >= 0)
                & (input_y < HEIGHT)
                & (input_x >= 0)
                & (input_x < WIDTH)
            )
            x = tl.load(
                x_ptr
                + ((batch * CHANNELS + channel) * HEIGHT + input_y) * WIDTH
                + input_x,
                mask=input_mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(
                weight_ptr
                + (
                    (output_channel * CHANNELS + channel) * KERNEL_HEIGHT
                    + kernel_y
                )
                * KERNEL_WIDTH
                + kernel_x,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += x * weight

    bias = tl.load(bias_ptr + output_channel).to(tl.float32)
    tl.store(out_ptr + output_index, tl.sum(accumulator, axis=0) + bias)


def op062_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """NCHW cross-correlation with a dense ``[C_out,C_in,K_h,K_w]`` filter."""
    require_visual_tensors(x, weight, bias)
    require_stride_padding(stride, padding)
    if x.ndim != 4 or weight.ndim != 4 or bias.ndim != 1:
        raise ValueError(
            "expected x[N,C,H,W], weight[C_out,C,K_h,K_w], and bias[C_out]"
        )
    batch, channels, height, width = x.shape
    out_channels, weight_channels, kernel_height, kernel_width = weight.shape
    if channels <= 0 or out_channels <= 0 or weight_channels != channels:
        raise ValueError("input and weight channels must be positive and match")
    if bias.shape != (out_channels,):
        raise ValueError("bias must have shape [C_out]")
    if kernel_height * kernel_width > MAX_DIRECT_KERNEL_ELEMENTS:
        raise ValueError(
            f"kernel area must not exceed {MAX_DIRECT_KERNEL_ELEMENTS}"
        )
    out_height = output_extent(height, kernel_height, stride, padding)
    out_width = output_extent(width, kernel_width, stride, padding)
    out = torch.empty(
        (batch, out_channels, out_height, out_width),
        device=x.device,
        dtype=torch.float32,
    )
    if out.numel():
        block_channels, warps = reduction_meta(channels, "input channels")
        _op062_conv2d_kernel[(out.numel(),)](
            x,
            weight,
            bias,
            out,
            CHANNELS=channels,
            HEIGHT=height,
            WIDTH=width,
            OUT_CHANNELS=out_channels,
            KERNEL_HEIGHT=kernel_height,
            KERNEL_WIDTH=kernel_width,
            OUT_HEIGHT=out_height,
            OUT_WIDTH=out_width,
            STRIDE=stride,
            PADDING=padding,
            BLOCK_CHANNELS=block_channels,
            num_warps=warps,
        )
    return out
