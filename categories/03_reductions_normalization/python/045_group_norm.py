"""045: group normalization."""

import math

import torch
import triton
import triton.language as tl

from ._common import (
    MAX_REDUCTION_SIZE,
    _reduction_meta,
    _require_cuda_contiguous,
    _require_float,
    _require_same_device,
    _require_same_dtype,
)


@triton.jit
def _op045_group_norm_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr, channels, spatial, groups, eps,
    BLOCK: tl.constexpr,
):
    program = tl.program_id(0)
    group = program % groups
    batch = program // groups
    channels_per_group = channels // groups
    count = channels_per_group * spatial
    offsets = tl.arange(0, BLOCK)
    mask = offsets < count
    base = batch * channels * spatial + group * count
    values = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(values, axis=0) / count
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / count
    channel = group * channels_per_group + offsets // spatial
    gamma = tl.load(gamma_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    result = centered * tl.rsqrt(variance + eps) * gamma + beta
    tl.store(out_ptr + base + offsets, result, mask=mask)


def op045_group_norm(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, groups: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    _require_cuda_contiguous(x, "x")
    _require_float(x, "x")
    _require_cuda_contiguous(gamma, "gamma")
    _require_cuda_contiguous(beta, "beta")
    _require_float(gamma, "gamma")
    _require_float(beta, "beta")
    _require_same_device(x, gamma, "gamma")
    _require_same_device(x, beta, "beta")
    _require_same_dtype(x, gamma, "gamma")
    _require_same_dtype(x, beta, "beta")
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, channels, spatial]")
    batch, channels, spatial = x.shape
    if groups <= 0 or channels == 0 or channels % groups != 0:
        raise ValueError("groups must be positive and divide channels")
    if gamma.shape != (channels,) or beta.shape != (channels,):
        raise ValueError("gamma and beta must have shape [channels]")
    count = channels // groups * spatial
    if count == 0 or count > MAX_REDUCTION_SIZE:
        raise ValueError(f"elements per group must be in [1, {MAX_REDUCTION_SIZE}]")
    if eps < 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and nonnegative")
    out = torch.empty_like(x)
    if batch:
        block, warps = _reduction_meta(count)
        _op045_group_norm_kernel[(batch * groups,)](
            x, gamma, beta, out, channels, spatial, groups, eps,
            BLOCK=block, num_warps=warps,
        )
    return out


__all__ = ["op045_group_norm"]
