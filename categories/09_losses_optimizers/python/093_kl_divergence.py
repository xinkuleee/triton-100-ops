"""093: row-wise KL(P || Q) from log probabilities."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import require_cuda_contiguous, require_floating, row_meta


@triton.jit
def _op093_kl_divergence_kernel(
    log_p_ptr, log_q_ptr, out_ptr, cols, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < cols
    log_p = tl.load(log_p_ptr + row * cols + col, mask=mask, other=0.0).to(tl.float32)
    log_q = tl.load(log_q_ptr + row * cols + col, mask=mask, other=0.0).to(tl.float32)
    p_is_zero = log_p == -float("inf")
    q_is_zero = log_q == -float("inf")
    finite_log_p = tl.where(p_is_zero, 0.0, log_p)
    finite_log_q = tl.where(q_is_zero, 0.0, log_q)
    finite_term = tl.exp(finite_log_p) * (finite_log_p - finite_log_q)
    nonzero_p_term = tl.where(q_is_zero, float("inf"), finite_term)
    term = tl.where(mask & (log_p != -float("inf")), nonzero_p_term, 0.0)
    tl.store(out_ptr + row, tl.sum(term, axis=0))


def op093_kl_divergence(
    log_p: torch.Tensor, log_q: torch.Tensor
) -> torch.Tensor:
    """Return row-wise ``KL(P || Q)`` from log probabilities."""
    require_cuda_contiguous(log_p, log_q)
    require_floating(log_p, log_q)
    if log_p.ndim != 2 or log_p.shape != log_q.shape:
        raise ValueError("log_p and log_q must have matching shape [rows, cols]")
    rows, cols = log_p.shape
    block, warps = row_meta(cols)
    out = torch.empty(rows, device=log_p.device, dtype=torch.float32)
    if rows:
        _op093_kl_divergence_kernel[(rows,)](
            log_p, log_q, out, cols, BLOCK=block, num_warps=warps
        )
    return out
