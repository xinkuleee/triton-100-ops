"""061: direct NCL one-dimensional cross-correlation."""

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
def _op061_conv1d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    LENGTH: tl.constexpr,
    OUT_CHANNELS: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    OUT_LENGTH: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    output_index = tl.program_id(0)
    output_x = output_index % OUT_LENGTH
    output_channel = (output_index // OUT_LENGTH) % OUT_CHANNELS
    batch = output_index // (OUT_CHANNELS * OUT_LENGTH)

    channel = tl.arange(0, BLOCK_CHANNELS)
    channel_mask = channel < CHANNELS
    accumulator = tl.zeros((BLOCK_CHANNELS,), tl.float32)
    for kernel_x in range(KERNEL_SIZE):
        input_x = output_x * STRIDE + kernel_x - PADDING
        input_mask = channel_mask & (input_x >= 0) & (input_x < LENGTH)
        x = tl.load(
            x_ptr + (batch * CHANNELS + channel) * LENGTH + input_x,
            mask=input_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr
            + (output_channel * CHANNELS + channel) * KERNEL_SIZE
            + kernel_x,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += x * weight

    bias = tl.load(bias_ptr + output_channel).to(tl.float32)
    tl.store(out_ptr + output_index, tl.sum(accumulator, axis=0) + bias)


def op061_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """NCL cross-correlation with weight ``[C_out, C_in, K]``."""
    require_visual_tensors(x, weight, bias)
    require_stride_padding(stride, padding)
    if x.ndim != 3 or weight.ndim != 3 or bias.ndim != 1:
        raise ValueError("expected x[N,C,L], weight[C_out,C,K], and bias[C_out]")
    batch, channels, length = x.shape
    out_channels, weight_channels, kernel_size = weight.shape
    if channels <= 0 or out_channels <= 0 or weight_channels != channels:
        raise ValueError("input and weight channels must be positive and match")
    if bias.shape != (out_channels,):
        raise ValueError("bias must have shape [C_out]")
    if kernel_size > MAX_DIRECT_KERNEL_ELEMENTS:
        raise ValueError(
            f"kernel size must not exceed {MAX_DIRECT_KERNEL_ELEMENTS}"
        )
    out_length = output_extent(length, kernel_size, stride, padding)
    out = torch.empty(
        (batch, out_channels, out_length), device=x.device, dtype=torch.float32
    )
    if out.numel():
        block_channels, warps = reduction_meta(channels, "input channels")
        _op061_conv1d_kernel[(out.numel(),)](
            x,
            weight,
            bias,
            out,
            CHANNELS=channels,
            LENGTH=length,
            OUT_CHANNELS=out_channels,
            KERNEL_SIZE=kernel_size,
            OUT_LENGTH=out_length,
            STRIDE=stride,
            PADDING=padding,
            BLOCK_CHANNELS=block_channels,
            num_warps=warps,
        )
    return out
