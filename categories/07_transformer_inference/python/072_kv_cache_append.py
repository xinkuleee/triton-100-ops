"""072: append key/value tensors to a logical KV cache."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import _require_cuda_contiguous, _require_floating


@triton.jit
def _op072_kv_cache_append_kernel(
    new_k_ptr, new_v_ptr, positions_ptr, cache_k_ptr, cache_v_ptr,
    HEADS: tl.constexpr, TOKENS: tl.constexpr, D: tl.constexpr,
    MAX_SEQUENCE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N
    dim = offset % D
    token = (offset // D) % TOKENS
    head = (offset // (D * TOKENS)) % HEADS
    batch = offset // (D * TOKENS * HEADS)
    position = tl.load(positions_ptr + batch * TOKENS + token, mask=valid, other=-1)
    valid &= (position >= 0) & (position < MAX_SEQUENCE)
    destination = ((batch * HEADS + head) * MAX_SEQUENCE + position) * D + dim
    tl.store(cache_k_ptr + destination, tl.load(new_k_ptr + offset, mask=valid), mask=valid)
    tl.store(cache_v_ptr + destination, tl.load(new_v_ptr + offset, mask=valid), mask=valid)


def op072_kv_cache_append(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    positions: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter new ``[B,H,T,D]`` keys/values into logical cache positions."""
    _require_cuda_contiguous(new_k, new_v, positions, cache_k, cache_v)
    _require_floating(new_k, new_v, cache_k, cache_v)
    if new_k.ndim != 4 or cache_k.ndim != 4:
        raise ValueError("new and cache tensors must be rank four")
    batch, heads, tokens, dim = new_k.shape
    max_sequence = cache_k.shape[2]
    if heads < 1 or dim < 1 or max_sequence < 1:
        raise ValueError("heads, dim, and cache max_sequence must be positive")
    if new_v.shape != new_k.shape or cache_v.shape != cache_k.shape:
        raise ValueError("key/value pairs must have matching shapes")
    if cache_k.shape != (batch, heads, max_sequence, dim):
        raise ValueError("cache must have shape [batch, heads, max_sequence, dim]")
    if positions.shape != (batch, tokens) or positions.dtype != torch.int64:
        raise ValueError("positions must be int64 with shape [batch, tokens]")
    if positions.numel():
        host_positions = positions.detach().cpu()
        if bool(((host_positions < 0) | (host_positions >= max_sequence)).any()):
            raise ValueError("positions must be valid cache indices")
        # Repeated positions in one batch would race across token writers.
        sorted_positions = torch.sort(host_positions, dim=1).values
        if tokens > 1 and bool((sorted_positions[:, 1:] == sorted_positions[:, :-1]).any()):
            raise ValueError("positions must be unique within each batch")
    count = new_k.numel()
    if count:
        _op072_kv_cache_append_kernel[(triton.cdiv(count, 256),)](
            new_k, new_v, positions, cache_k, cache_v, HEADS=heads, TOKENS=tokens,
            D=dim, MAX_SEQUENCE=max_sequence, N=count, BLOCK=256, num_warps=4,
        )
    return cache_k, cache_v
