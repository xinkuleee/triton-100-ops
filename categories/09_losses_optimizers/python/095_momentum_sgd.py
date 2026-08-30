"""095: in-place momentum SGD."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ._common import (
    require_cuda_contiguous,
    require_floating,
    require_matching_shape,
    validate_learning_rate,
)


@triton.jit
def _op095_momentum_sgd_kernel(
    param_ptr, grad_ptr, velocity_ptr, learning_rate, momentum, count,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    param = tl.load(param_ptr + offset, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offset, mask=mask).to(tl.float32)
    velocity = momentum * tl.load(velocity_ptr + offset, mask=mask).to(tl.float32) + grad
    tl.store(velocity_ptr + offset, velocity, mask=mask)
    tl.store(param_ptr + offset, param - learning_rate * velocity, mask=mask)


def op095_momentum_sgd_(
    param: torch.Tensor, grad: torch.Tensor, velocity: torch.Tensor,
    learning_rate: float, momentum: float = 0.9,
) -> torch.Tensor:
    require_cuda_contiguous(param, grad, velocity)
    require_floating(param, grad, velocity)
    require_matching_shape(param, grad, velocity)
    validate_learning_rate(learning_rate)
    if momentum < 0.0 or not math.isfinite(momentum):
        raise ValueError("momentum must be finite and nonnegative")
    count = param.numel()
    if count:
        _op095_momentum_sgd_kernel[(triton.cdiv(count, 256),)](
            param, grad, velocity, learning_rate, momentum, count,
            BLOCK=256, num_warps=4,
        )
    return param
