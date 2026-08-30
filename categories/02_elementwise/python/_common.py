"""Shared validation, launch geometry, and numerical helpers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 256
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def stable_sigmoid(value):
    exponential = tl.exp(-tl.abs(value))
    return tl.where(
        value >= 0.0,
        1.0 / (1.0 + exponential),
        exponential / (1.0 + exponential),
    )


@triton.jit
def stable_softplus(value):
    return tl.maximum(value, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(value)))


def require_float_tensor(x: torch.Tensor, name: str) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError(f"{name} must be on a CUDA or ROCm device")
    if x.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{name} must have dtype float16, bfloat16, or float32")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def require_pair(x: torch.Tensor, y: torch.Tensor) -> None:
    require_float_tensor(x, "x")
    require_float_tensor(y, "y")
    if x.shape != y.shape or x.device != y.device or x.dtype != y.dtype:
        raise ValueError("x and y must have the same shape, device, and dtype")


def grid(n_elements: int) -> tuple[int]:
    return (triton.cdiv(n_elements, BLOCK_SIZE),)
