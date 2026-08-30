"""091: class-index cross entropy with fused log-softmax."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import require_cuda_contiguous, require_floating, row_meta


@triton.jit
def _op091_cross_entropy_kernel(
    logits_ptr, labels_ptr, out_ptr, cols, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < cols
    logits = tl.load(
        logits_ptr + row * cols + col, mask=mask, other=-float("inf")
    ).to(tl.float32)
    maximum = tl.max(logits, axis=0)
    log_partition = maximum + tl.log(tl.sum(tl.exp(logits - maximum), axis=0))
    label = tl.load(labels_ptr + row)
    selected = tl.sum(tl.where(col == label, logits, 0.0), axis=0)
    tl.store(out_ptr + row, log_partition - selected)


def op091_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Return one unreduced class-index cross-entropy value per row."""
    require_cuda_contiguous(logits, labels)
    require_floating(logits)
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("expected logits[rows, classes] and labels[rows]")
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError("labels must be int32 or int64")
    rows, cols = logits.shape
    if rows and bool(((labels < 0) | (labels >= cols)).any().item()):
        raise ValueError("every label must be in [0, classes)")
    block, warps = row_meta(cols)
    out = torch.empty(rows, device=logits.device, dtype=torch.float32)
    if rows:
        _op091_cross_entropy_kernel[(rows,)](
            logits, labels, out, cols, BLOCK=block, num_warps=warps
        )
    return out
