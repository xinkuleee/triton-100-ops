"""090: numerically stable binary cross entropy with logits."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import require_cuda_contiguous, require_floating, require_matching_shape


@triton.jit
def _op090_bce_with_logits_kernel(
    logits_ptr, targets_ptr, out_ptr, count, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    logits = tl.load(logits_ptr + offset, mask=mask).to(tl.float32)
    targets = tl.load(targets_ptr + offset, mask=mask).to(tl.float32)
    loss = (
        tl.maximum(logits, 0.0) - logits * targets
        + tl.log(1.0 + tl.exp(-tl.abs(logits)))
    )
    tl.store(out_ptr + offset, loss, mask=mask)


def op090_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Return unreduced, overflow-stable binary cross entropy."""
    require_cuda_contiguous(logits, targets)
    require_floating(logits, targets)
    require_matching_shape(logits, targets)
    out = torch.empty_like(logits, dtype=torch.float32)
    count = logits.numel()
    if count:
        _op090_bce_with_logits_kernel[(triton.cdiv(count, 256),)](
            logits, targets, out, count, BLOCK=256, num_warps=4
        )
    return out
