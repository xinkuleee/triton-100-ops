"""064: one-by-one NCHW convolution."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import reduction_meta, require_visual_tensors


@triton.jit
def _op064_pointwise_conv2d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    OUT_CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    output_index = tl.program_id(0)
    pixel = output_index % SPATIAL
    output_channel = (output_index // SPATIAL) % OUT_CHANNELS
    batch = output_index // (OUT_CHANNELS * SPATIAL)
    channel = tl.arange(0, BLOCK_CHANNELS)
    channel_mask = channel < CHANNELS
    x = tl.load(
        x_ptr + (batch * CHANNELS + channel) * SPATIAL + pixel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(
        weight_ptr + output_channel * CHANNELS + channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(bias_ptr + output_channel).to(tl.float32)
    tl.store(out_ptr + output_index, tl.sum(x * weight, axis=0) + bias)


def op064_pointwise_conv2d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """One-by-one NCHW convolution with ``weight[C_out,C_in]``."""
    require_visual_tensors(x, weight, bias)
    if x.ndim != 4 or weight.ndim not in (2, 4) or bias.ndim != 1:
        raise ValueError(
            "expected x[N,C,H,W], weight[C_out,C] or [C_out,C,1,1], bias[C_out]"
        )
    batch, channels, height, width = x.shape
    out_channels = weight.shape[0]
    if channels <= 0 or height <= 0 or width <= 0 or out_channels <= 0:
        raise ValueError("channels and spatial input dimensions must be positive")
    if weight.ndim == 2:
        valid_weight = weight.shape[1] == channels
    else:
        valid_weight = weight.shape[1:] == (channels, 1, 1)
    if not valid_weight or bias.shape != (out_channels,):
        raise ValueError("pointwise weight or bias shape does not match x")
    out = torch.empty(
        (batch, out_channels, height, width),
        device=x.device,
        dtype=torch.float32,
    )
    if out.numel():
        block_channels, warps = reduction_meta(channels, "input channels")
        flat_weight = weight.reshape(out_channels, channels)
        _op064_pointwise_conv2d_kernel[(out.numel(),)](
            x,
            flat_weight,
            bias,
            out,
            CHANNELS=channels,
            OUT_CHANNELS=out_channels,
            SPATIAL=height * width,
            BLOCK_CHANNELS=block_channels,
            num_warps=warps,
        )
    return out
