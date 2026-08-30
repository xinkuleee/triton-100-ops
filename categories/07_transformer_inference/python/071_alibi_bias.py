"""071: ALiBi attention-score bias."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import _require_cuda_contiguous, _require_floating


@triton.jit
def _op071_alibi_bias_kernel(
    scores_ptr, slopes_ptr, out_ptr, HEADS: tl.constexpr, QUERIES: tl.constexpr,
    KEYS: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N
    key = offset % KEYS
    query = (offset // KEYS) % QUERIES
    head = (offset // (KEYS * QUERIES)) % HEADS
    score = tl.load(scores_ptr + offset, mask=valid)
    slope = tl.load(slopes_ptr + head, mask=valid)
    biased = score + slope * (key - query).to(tl.float32)
    tl.store(out_ptr + offset, biased, mask=valid)


def op071_alibi_bias(scores: torch.Tensor, slopes: torch.Tensor) -> torch.Tensor:
    """Add ``slope[h] * (key - query)`` to scores ``[B,H,Q,K]``."""
    _require_cuda_contiguous(scores, slopes)
    _require_floating(scores, slopes)
    if scores.ndim != 4:
        raise ValueError("scores must have shape [batch, heads, queries, keys]")
    _, heads, queries, keys = scores.shape
    if heads < 1 or queries < 1 or keys < 1:
        raise ValueError("heads, queries, and keys must be positive")
    if slopes.shape != (heads,):
        raise ValueError("slopes must have shape [heads]")
    out = torch.empty_like(scores)
    count = scores.numel()
    if count:
        _op071_alibi_bias_kernel[(triton.cdiv(count, 256),)](
            scores, slopes, out, HEADS=heads, QUERIES=queries, KEYS=keys,
            N=count, BLOCK=256, num_warps=4,
        )
    return out
