"""080: greedy decoding by row-wise argmax."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import _require_cuda_contiguous, _require_floating, _row_meta


@triton.jit
def _op080_greedy_decode_kernel(logits_ptr, out_ptr, vocab, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    token = tl.arange(0, BLOCK)
    logits = tl.load(
        logits_ptr + batch * vocab + token, mask=token < vocab, other=-float("inf")
    ).to(tl.float32)
    tl.store(out_ptr + batch, tl.argmax(logits, axis=0, tie_break_left=True))


def op080_greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    _require_cuda_contiguous(logits)
    _require_floating(logits)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab]")
    batch, vocab = logits.shape
    block, warps = _row_meta(vocab)
    out = torch.empty(batch, device=logits.device, dtype=torch.int64)
    if batch:
        _op080_greedy_decode_kernel[(batch,)](
            logits, out, vocab, BLOCK=block, num_warps=warps
        )
    return out
