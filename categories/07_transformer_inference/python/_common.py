"""Shared validation and launch limits for transformer-inference kernels."""

from __future__ import annotations

import torch
import triton


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_ROW = 65_536


def _require_cuda_contiguous(*tensors: torch.Tensor) -> None:
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("all inputs must be torch.Tensor instances")
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all tensors must be contiguous CUDA/ROCm tensors")
    if tensors and any(tensor.device != tensors[0].device for tensor in tensors[1:]):
        raise ValueError("all tensors must be on the same device")


def _require_floating(*tensors: torch.Tensor) -> None:
    if any(tensor.dtype not in _FLOAT_DTYPES for tensor in tensors):
        raise TypeError("floating tensors must use float16, bfloat16, or float32")
    if tensors and any(tensor.dtype != tensors[0].dtype for tensor in tensors[1:]):
        raise TypeError("floating tensors must have the same dtype")


def _row_meta(width: int) -> tuple[int, int]:
    if width < 1 or width > _MAX_ROW:
        raise ValueError(f"row width must be in [1, {_MAX_ROW}]")
    block = triton.next_power_of_2(width)
    return block, 4 if block <= 2_048 else 8
