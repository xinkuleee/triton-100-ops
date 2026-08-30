"""044: batch normalization in inference mode."""

import math

import torch
import triton
import triton.language as tl

from ._common import (
    _require_cuda_contiguous,
    _require_float,
    _require_matrix,
    _require_same_device,
    _require_same_dtype,
)


@triton.jit
def _op044_batch_norm_inference_kernel(
    x_ptr, mean_ptr, variance_ptr, gamma_ptr, beta_ptr, out_ptr, total, cols, eps,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    channels = offsets % cols
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=mask).to(tl.float32)
    variance = tl.load(variance_ptr + channels, mask=mask).to(tl.float32)
    gamma = tl.load(gamma_ptr + channels, mask=mask).to(tl.float32)
    beta = tl.load(beta_ptr + channels, mask=mask).to(tl.float32)
    result = (x - mean) * tl.rsqrt(variance + eps) * gamma + beta
    tl.store(out_ptr + offsets, result, mask=mask)


def op044_batch_norm_inference(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_variance: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    rows, cols = _require_matrix(x)
    for tensor, name in (
        (running_mean, "running_mean"),
        (running_variance, "running_variance"),
        (gamma, "gamma"),
        (beta, "beta"),
    ):
        _require_cuda_contiguous(tensor, name)
        _require_float(tensor, name)
        _require_same_device(x, tensor, name)
        _require_same_dtype(x, tensor, name)
        if tensor.shape != (cols,):
            raise ValueError(f"{name} must have shape [cols]")
    if eps < 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and nonnegative")
    out = torch.empty_like(x)
    total = rows * cols
    if total:
        grid = (triton.cdiv(total, 256),)
        _op044_batch_norm_inference_kernel[grid](
            x, running_mean, running_variance, gamma, beta, out, total, cols, eps,
            BLOCK=256,
        )
    return out


__all__ = ["op044_batch_norm_inference"]
