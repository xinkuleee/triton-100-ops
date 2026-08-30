"""Shared tensor validation and linear-algebra size limits."""

from __future__ import annotations

import torch


FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MAX_DOT_K = 65_536
MAX_GEMV_K = 65_536


def require_tensors(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("inputs must be torch.Tensor objects")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("inputs must be contiguous CUDA or ROCm tensors")
        if tensor.dtype not in FLOAT_DTYPES:
            raise TypeError("inputs must use float16, bfloat16, or float32")
    if tensors and any(t.device != tensors[0].device for t in tensors[1:]):
        raise ValueError("inputs must be on the same device")
    if tensors and any(t.dtype != tensors[0].dtype for t in tensors[1:]):
        raise ValueError("inputs must have the same dtype")
