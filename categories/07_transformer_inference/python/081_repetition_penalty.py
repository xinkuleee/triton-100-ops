"""081: in-place repetition penalty for decoded token histories."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ._common import _require_cuda_contiguous, _require_floating, _row_meta


@triton.jit
def _op081_repetition_penalty_kernel(
    logits_ptr, token_ids_ptr, penalty, history, vocab, BLOCK: tl.constexpr
):
    batch = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    mask = offset < history
    token = tl.load(token_ids_ptr + batch * history + offset, mask=mask, other=0)
    valid = mask & (token >= 0) & (token < vocab)
    value = tl.load(logits_ptr + batch * vocab + token, mask=valid, other=0.0)
    adjusted = tl.where(value > 0.0, value / penalty, value * penalty)
    tl.store(logits_ptr + batch * vocab + token, adjusted, mask=valid)


def op081_repetition_penalty_(
    logits: torch.Tensor, token_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Modify logits in place; token IDs must be unique within each batch row."""
    _require_cuda_contiguous(logits, token_ids)
    _require_floating(logits)
    if logits.ndim != 2 or token_ids.ndim != 2 or logits.shape[0] != token_ids.shape[0]:
        raise ValueError("expected logits[B,V] and token_ids[B,history]")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("token_ids must be int32 or int64")
    if penalty <= 0.0 or not math.isfinite(penalty):
        raise ValueError("penalty must be finite and positive")
    batch, vocab = logits.shape
    history = token_ids.shape[1]
    if vocab < 1:
        raise ValueError("vocab must be positive")
    if not history or not batch:
        return logits
    invalid = (token_ids < 0) | (token_ids >= vocab)
    if bool(invalid.any().item()):
        raise ValueError("token_ids must be in [0, vocab)")
    # Duplicate destinations would race in the parallel in-place kernel.
    sorted_ids = torch.sort(token_ids, dim=1).values
    if history > 1 and bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any().item()):
        raise ValueError("token_ids must be unique within every batch row")
    block, warps = _row_meta(history)
    _op081_repetition_penalty_kernel[(batch,)](
        logits, token_ids, penalty, history, vocab, BLOCK=block, num_warps=warps
    )
    return logits
