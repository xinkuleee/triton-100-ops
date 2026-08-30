"""Shared validation and launch helpers for operators 034--046."""

import torch
import triton


MAX_REDUCTION_SIZE = 65_536
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _require_cuda_contiguous(x: torch.Tensor, name: str) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError(f"{name} must be on a CUDA or ROCm device")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_float(x: torch.Tensor, name: str) -> None:
    if x.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{name} must have dtype float16, bfloat16, or float32")


def _require_same_device(reference: torch.Tensor, other: torch.Tensor, name: str) -> None:
    if reference.device != other.device:
        raise ValueError(f"{name} must be on the same device as x")


def _require_same_dtype(reference: torch.Tensor, other: torch.Tensor, name: str) -> None:
    if reference.dtype != other.dtype:
        raise TypeError(f"{name} must have the same dtype as x")


def _require_matrix(x: torch.Tensor, name: str = "x") -> tuple[int, int]:
    _require_cuda_contiguous(x, name)
    _require_float(x, name)
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [rows, cols]")
    rows, cols = x.shape
    if cols == 0 or cols > MAX_REDUCTION_SIZE:
        raise ValueError(f"{name}.shape[1] must be in [1, {MAX_REDUCTION_SIZE}]")
    return rows, cols


def _reduction_meta(cols: int) -> tuple[int, int]:
    block = triton.next_power_of_2(cols)
    warps = 4 if block <= 2_048 else 8
    return block, warps

