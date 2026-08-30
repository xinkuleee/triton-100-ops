"""070: interleaved rotary positional embeddings."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import _require_cuda_contiguous, _require_floating


@triton.jit
def _op070_rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr, HEADS: tl.constexpr, D: tl.constexpr,
    PAIRS: tl.constexpr, BLOCK: tl.constexpr,
):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    half = D // 2
    valid = pair < PAIRS
    frequency = pair % half
    head = (pair // half) % HEADS
    token = pair // (half * HEADS)
    base = (token * HEADS + head) * D + 2 * frequency
    even = tl.load(x_ptr + base, mask=valid)
    odd = tl.load(x_ptr + base + 1, mask=valid)
    cosine = tl.load(cos_ptr + token * half + frequency, mask=valid)
    sine = tl.load(sin_ptr + token * half + frequency, mask=valid)
    tl.store(out_ptr + base, even * cosine - odd * sine, mask=valid)
    tl.store(out_ptr + base + 1, even * sine + odd * cosine, mask=valid)


def op070_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply interleaved even/odd rotary embeddings to ``x[T,H,D]``."""
    _require_cuda_contiguous(x, cos, sin)
    _require_floating(x, cos, sin)
    if x.ndim != 3:
        raise ValueError("x must have shape [tokens, heads, dim]")
    tokens, heads, dim = x.shape
    if heads < 1 or dim < 2 or dim % 2:
        raise ValueError("head dimension must be positive and even")
    if cos.shape != (tokens, dim // 2) or sin.shape != cos.shape:
        raise ValueError("cos and sin must have shape [tokens, dim / 2]")
    out = torch.empty_like(x)
    pairs = tokens * heads * (dim // 2)
    if pairs:
        _op070_rope_kernel[(triton.cdiv(pairs, 256),)](
            x, cos, sin, out, HEADS=heads, D=dim, PAIRS=pairs, BLOCK=256, num_warps=4
        )
    return out
