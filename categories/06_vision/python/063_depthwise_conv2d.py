"""063: depthwise NCHW two-dimensional cross-correlation."""

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
def _op063_depthwise_conv2d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    KERNEL_HEIGHT: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    OUT_HEIGHT: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    BLOCK_KERNEL: tl.constexpr,
):
    output_index = tl.program_id(0)
    output_x = output_index % OUT_WIDTH
    output_y = (output_index // OUT_WIDTH) % OUT_HEIGHT
    channel = (output_index // (OUT_WIDTH * OUT_HEIGHT)) % CHANNELS
    batch = output_index // (CHANNELS * OUT_HEIGHT * OUT_WIDTH)

    kernel_offset = tl.arange(0, BLOCK_KERNEL)
    kernel_y = kernel_offset // KERNEL_WIDTH
    kernel_x = kernel_offset % KERNEL_WIDTH
    input_y = output_y * STRIDE + kernel_y - PADDING
    input_x = output_x * STRIDE + kernel_x - PADDING
    kernel_mask = kernel_offset < KERNEL_HEIGHT * KERNEL_WIDTH
    input_mask = (
        kernel_mask
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
        weight_ptr + channel * KERNEL_HEIGHT * KERNEL_WIDTH + kernel_offset,
        mask=kernel_mask,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(bias_ptr + channel).to(tl.float32)
    tl.store(out_ptr + output_index, tl.sum(x * weight, axis=0) + bias)


def op063_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """Depthwise NCHW cross-correlation with multiplier one (``groups=C``)."""
    require_visual_tensors(x, weight, bias)
    require_stride_padding(stride, padding)
    if x.ndim != 4 or weight.ndim not in (3, 4) or bias.ndim != 1:
        raise ValueError(
            "expected x[N,C,H,W], weight[C,K_h,K_w] or [C,1,K_h,K_w], bias[C]"
        )
    batch, channels, height, width = x.shape
    if channels <= 0 or weight.shape[0] != channels or bias.shape != (channels,):
        raise ValueError("depthwise channels must be positive and match")
    if weight.ndim == 4 and weight.shape[1] != 1:
        raise ValueError(
            "four-dimensional depthwise weight must have shape [C,1,K_h,K_w]"
        )
    kernel_height, kernel_width = weight.shape[-2:]
    kernel_elements = kernel_height * kernel_width
    if kernel_elements > MAX_DIRECT_KERNEL_ELEMENTS:
        raise ValueError(
            f"kernel area must not exceed {MAX_DIRECT_KERNEL_ELEMENTS}"
        )
    out_height = output_extent(height, kernel_height, stride, padding)
    out_width = output_extent(width, kernel_width, stride, padding)
    out = torch.empty(
        (batch, channels, out_height, out_width),
        device=x.device,
        dtype=torch.float32,
    )
    if out.numel():
        block_kernel, warps = reduction_meta(kernel_elements, "kernel area")
        flat_weight = weight.reshape(channels, kernel_height, kernel_width)
        _op063_depthwise_conv2d_kernel[(out.numel(),)](
            x,
            flat_weight,
            bias,
            out,
            CHANNELS=channels,
            HEIGHT=height,
            WIDTH=width,
            KERNEL_HEIGHT=kernel_height,
            KERNEL_WIDTH=kernel_width,
            OUT_HEIGHT=out_height,
            OUT_WIDTH=out_width,
            STRIDE=stride,
            PADDING=padding,
            BLOCK_KERNEL=block_kernel,
            num_warps=warps,
        )
    return out
