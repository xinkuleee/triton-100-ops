"""098: in-place Adagrad update."""

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
def _op098_adagrad_kernel(
    param_ptr, grad_ptr, state_ptr, learning_rate, eps, count, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    param = tl.load(param_ptr + offset, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offset, mask=mask).to(tl.float32)
    state = tl.load(state_ptr + offset, mask=mask).to(tl.float32) + grad * grad
    tl.store(state_ptr + offset, state, mask=mask)
    tl.store(
        param_ptr + offset, param - learning_rate * grad / (tl.sqrt(state) + eps),
        mask=mask,
    )


def op098_adagrad_(
    param: torch.Tensor, grad: torch.Tensor, state: torch.Tensor,
    learning_rate: float = 1e-2, eps: float = 1e-10,
) -> torch.Tensor:
    require_cuda_contiguous(param, grad, state)
    require_floating(param, grad, state)
    require_matching_shape(param, grad, state)
    validate_learning_rate(learning_rate)
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    count = param.numel()
    if count:
        _op098_adagrad_kernel[(triton.cdiv(count, 256),)](
            param, grad, state, learning_rate, eps, count, BLOCK=256, num_warps=4
        )
    return param
