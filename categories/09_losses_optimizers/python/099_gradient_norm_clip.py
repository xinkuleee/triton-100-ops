"""099: in-place gradient norm clipping."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ._common import require_cuda_contiguous, require_floating


@triton.jit
def _op099_gradient_norm_clip_kernel(
    grad_ptr, norm, max_norm, eps, count, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    scale = tl.minimum(1.0, max_norm / (norm + eps))
    grad = tl.load(grad_ptr + offset, mask=mask).to(tl.float32)
    tl.store(grad_ptr + offset, grad * scale, mask=mask)


def op099_gradient_norm_clip_(
    grad: torch.Tensor, norm: float, max_norm: float, eps: float = 1e-6
) -> torch.Tensor:
    """Scale ``grad`` in place using a norm computed by an earlier reduction."""
    require_cuda_contiguous(grad)
    require_floating(grad)
    if (
        not math.isfinite(norm)
        or not math.isfinite(max_norm)
        or not math.isfinite(eps)
        or norm < 0.0
        or max_norm < 0.0
        or eps <= 0.0
    ):
        raise ValueError(
            "norm/max_norm must be finite and nonnegative; eps must be finite and positive"
        )
    count = grad.numel()
    if count:
        _op099_gradient_norm_clip_kernel[(triton.cdiv(count, 256),)](
            grad, norm, max_norm, eps, count, BLOCK=256, num_warps=4
        )
    return grad
