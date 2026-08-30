"""094: in-place stochastic gradient descent."""

from __future__ import annotations

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
def _op094_sgd_kernel(param_ptr, grad_ptr, learning_rate, count, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    param = tl.load(param_ptr + offset, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offset, mask=mask).to(tl.float32)
    tl.store(param_ptr + offset, param - learning_rate * grad, mask=mask)


def op094_sgd_(
    param: torch.Tensor, grad: torch.Tensor, learning_rate: float
) -> torch.Tensor:
    require_cuda_contiguous(param, grad)
    require_floating(param, grad)
    require_matching_shape(param, grad)
    validate_learning_rate(learning_rate)
    count = param.numel()
    if count:
        _op094_sgd_kernel[(triton.cdiv(count, 256),)](
            param, grad, learning_rate, count, BLOCK=256, num_warps=4
        )
    return param
