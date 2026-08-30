"""100: detect NaN or infinity in a floating tensor."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import require_cuda_contiguous, require_floating


@triton.jit
def _op100_non_finite_check_kernel(
    x_ptr, flag_ptr, count, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    value = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    bad_count = tl.sum(
        tl.where(mask & ((value != value) | (tl.abs(value) == float("inf"))), 1, 0),
        axis=0,
    )
    tl.atomic_or(flag_ptr, 1, mask=bad_count > 0)


def op100_non_finite_check(x: torch.Tensor) -> torch.Tensor:
    """Return a device int32 scalar: one if any value is NaN/Inf, otherwise zero."""
    require_cuda_contiguous(x)
    require_floating(x)
    flag = torch.zeros((), device=x.device, dtype=torch.int32)
    count = x.numel()
    if count:
        _op100_non_finite_check_kernel[(triton.cdiv(count, 256),)](
            x, flag, count, BLOCK=256, num_warps=4
        )
    return flag
