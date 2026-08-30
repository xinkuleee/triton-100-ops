"""Shared validation and launch geometry for operators 047--056."""

from __future__ import annotations

import torch
import triton


FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MAX_ROW = 65_536
MAX_BAG = 4_096


def require_device_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA or ROCm tensor")


def require_float_tensor(tensor: torch.Tensor, name: str) -> None:
    require_device_tensor(tensor, name)
    if tensor.dtype not in FLOAT_DTYPES:
        raise TypeError(f"{name} must use float16, bfloat16, or float32")


def require_same_device(reference: torch.Tensor, *others: torch.Tensor) -> None:
    if any(tensor.device != reference.device for tensor in others):
        raise ValueError("all tensors must be on the same device")


def row_launch_meta(width: int) -> tuple[int, int]:
    if width < 1 or width > MAX_ROW:
        raise ValueError(f"row width must be in [1, {MAX_ROW}]")
    block = triton.next_power_of_2(width)
    return block, 4 if block <= 2_048 else 8
