"""Shared validation helpers for loss and optimizer operators 089--100."""

from __future__ import annotations

import math

import torch
import triton

FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MAX_ROW = 65_536


def require_cuda_contiguous(*tensors: torch.Tensor) -> None:
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("all inputs must be torch.Tensor instances")
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all tensors must be contiguous CUDA/ROCm tensors")
    if tensors and any(tensor.device != tensors[0].device for tensor in tensors[1:]):
        raise ValueError("all tensors must be on the same device")


def require_floating(*tensors: torch.Tensor) -> None:
    if any(tensor.dtype not in FLOAT_DTYPES for tensor in tensors):
        raise TypeError("floating tensors must use float16, bfloat16, or float32")
    if tensors and any(tensor.dtype != tensors[0].dtype for tensor in tensors[1:]):
        raise TypeError("floating tensors must have the same dtype")


def require_matching_shape(*tensors: torch.Tensor) -> None:
    if tensors and any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
        raise ValueError("tensor shapes must match")


def row_meta(cols: int) -> tuple[int, int]:
    if cols < 1 or cols > MAX_ROW:
        raise ValueError(f"row width must be in [1, {MAX_ROW}]")
    block = triton.next_power_of_2(cols)
    return block, 4 if block <= 2_048 else 8


def validate_learning_rate(learning_rate: float) -> None:
    if learning_rate < 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and nonnegative")
