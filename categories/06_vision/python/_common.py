"""Shared validation and launch metadata for vision operators."""

from __future__ import annotations

import torch
import triton


FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MAX_REDUCTION_WIDTH = 65_536
MAX_DIRECT_KERNEL_ELEMENTS = 4_096
RESIZE_BLOCK = 256


def require_visual_tensors(*tensors: torch.Tensor) -> None:
    if not tensors:
        return
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("inputs must be torch.Tensor objects")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("inputs must be contiguous CUDA or ROCm tensors")
        if tensor.dtype not in FLOAT_DTYPES:
            raise TypeError("inputs must use float16, bfloat16, or float32")
    if any(tensor.device != tensors[0].device for tensor in tensors[1:]):
        raise ValueError("inputs must be on the same device")
    if any(tensor.dtype != tensors[0].dtype for tensor in tensors[1:]):
        raise TypeError("inputs must have the same dtype")


def require_stride_padding(stride: int, padding: int) -> None:
    if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
        raise ValueError("padding must be a nonnegative integer")


def output_extent(
    input_size: int, kernel_size: int, stride: int, padding: int
) -> int:
    numerator = input_size + 2 * padding - kernel_size
    if input_size <= 0 or kernel_size <= 0 or numerator < 0:
        raise ValueError("kernel must fit the padded, nonempty input")
    return numerator // stride + 1


def kernel_pair(kernel: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(kernel, int) and not isinstance(kernel, bool):
        kernel = (kernel, kernel)
    if (
        not isinstance(kernel, tuple)
        or len(kernel) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) for value in kernel
        )
        or kernel[0] <= 0
        or kernel[1] <= 0
    ):
        raise ValueError("kernel must be a positive integer or a pair of integers")
    return kernel


def reduction_meta(width: int, label: str) -> tuple[int, int]:
    if width <= 0 or width > MAX_REDUCTION_WIDTH:
        raise ValueError(f"{label} must be in [1, {MAX_REDUCTION_WIDTH}]")
    block = triton.next_power_of_2(width)
    return block, 4 if block <= 2_048 else 8
