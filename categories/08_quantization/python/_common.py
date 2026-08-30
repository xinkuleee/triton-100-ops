"""Shared validation, size limits, and symmetric INT8 rounding."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MAX_REDUCTION_SIZE = 65_536
MAX_SAFE_INT8_DOT_K = 131_071


def _require_cuda_contiguous(*tensors: torch.Tensor) -> None:
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("all inputs must be torch.Tensor instances")
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all tensors must be contiguous CUDA/ROCm tensors")
    if tensors and any(tensor.device != tensors[0].device for tensor in tensors[1:]):
        raise ValueError("all tensors must be on the same device")


def _require_floating(*tensors: torch.Tensor) -> None:
    if any(tensor.dtype not in FLOAT_DTYPES for tensor in tensors):
        raise TypeError("floating tensors must use float16, bfloat16, or float32")


def _require_scalar_scale(scale: torch.Tensor) -> None:
    if scale.numel() != 1:
        raise ValueError("scale must contain exactly one value")
    value = float(scale.detach().cpu().item())
    if not (value > 0.0 and torch.isfinite(torch.tensor(value))):
        raise ValueError("scale must be finite and strictly positive")


def _reduction_meta(width: int) -> tuple[int, int]:
    if width < 1 or width > MAX_REDUCTION_SIZE:
        raise ValueError(f"reduction width must be in [1, {MAX_REDUCTION_SIZE}]")
    block = triton.next_power_of_2(width)
    return block, 4 if block <= 2_048 else 8


@triton.jit
def _op082_round_and_clamp_int8(value):
    clipped = tl.maximum(-127.0, tl.minimum(127.0, value))
    rounded = tl.where(
        clipped >= 0.0, tl.floor(clipped + 0.5), tl.ceil(clipped - 0.5)
    )
    return rounded.to(tl.int8)
