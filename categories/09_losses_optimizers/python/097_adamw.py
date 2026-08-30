"""097: in-place AdamW update."""

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
def _op097_adamw_kernel(
    param_ptr, grad_ptr, first_ptr, second_ptr, learning_rate, beta1, beta2,
    eps, weight_decay, correction1, correction2, count, BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    param = tl.load(param_ptr + offset, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offset, mask=mask).to(tl.float32)
    first = beta1 * tl.load(first_ptr + offset, mask=mask).to(tl.float32) + (1.0 - beta1) * grad
    second = beta2 * tl.load(second_ptr + offset, mask=mask).to(tl.float32) + (1.0 - beta2) * grad * grad
    update = (first / correction1) / (tl.sqrt(second / correction2) + eps)
    update += weight_decay * param
    tl.store(first_ptr + offset, first, mask=mask)
    tl.store(second_ptr + offset, second, mask=mask)
    tl.store(param_ptr + offset, param - learning_rate * update, mask=mask)


def op097_adamw_(
    param: torch.Tensor, grad: torch.Tensor, first_moment: torch.Tensor,
    second_moment: torch.Tensor, step: int, learning_rate: float = 1e-3,
    beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
    weight_decay: float = 1e-2,
) -> torch.Tensor:
    require_cuda_contiguous(param, grad, first_moment, second_moment)
    require_floating(param, grad, first_moment, second_moment)
    require_matching_shape(param, grad, first_moment, second_moment)
    validate_learning_rate(learning_rate)
    if (
        step < 1
        or not math.isfinite(beta1)
        or not math.isfinite(beta2)
        or not math.isfinite(eps)
        or not math.isfinite(weight_decay)
        or not 0.0 <= beta1 < 1.0
        or not 0.0 <= beta2 < 1.0
        or eps <= 0.0
        or weight_decay < 0.0
    ):
        raise ValueError("invalid AdamW step or hyperparameter")
    correction1, correction2 = 1.0 - beta1 ** step, 1.0 - beta2 ** step
    count = param.numel()
    if count:
        _op097_adamw_kernel[(triton.cdiv(count, 256),)](
            param, grad, first_moment, second_moment, learning_rate, beta1, beta2,
            eps, weight_decay, correction1, correction2, count, BLOCK=256, num_warps=4,
        )
    return param
