"""073: attention over a paged key/value cache."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import _require_cuda_contiguous, _require_floating


@triton.jit
def _op073_paged_attention_kernel(
    q_ptr, key_cache_ptr, value_cache_ptr, table_ptr, lengths_ptr, out_ptr,
    HEADS: tl.constexpr, D: tl.constexpr, PAGE_SIZE: tl.constexpr,
    MAX_PAGES: tl.constexpr, MAX_SEQUENCE: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    batch_head = tl.program_id(0)
    batch = batch_head // HEADS
    head = batch_head % HEADS
    token_offsets = tl.arange(0, BLOCK_T)
    dim_offsets = tl.arange(0, BLOCK_D)
    raw_length = tl.load(lengths_ptr + batch)
    length = tl.minimum(tl.maximum(raw_length, 0), MAX_SEQUENCE)
    valid_token = token_offsets < length
    physical_page = tl.load(
        table_ptr + batch * MAX_PAGES + token_offsets // PAGE_SIZE,
        mask=valid_token, other=0,
    )
    cache_base = (
        (physical_page * PAGE_SIZE + token_offsets % PAGE_SIZE) * HEADS + head
    ) * D
    q = tl.load(
        q_ptr + batch_head * D + dim_offsets, mask=dim_offsets < D, other=0.0
    ).to(tl.float32)
    cache_mask = valid_token[:, None] & (dim_offsets[None, :] < D)
    k = tl.load(
        key_cache_ptr + cache_base[:, None] + dim_offsets[None, :],
        mask=cache_mask, other=0.0,
    ).to(tl.float32)
    score = tl.sum(k * q[None, :], axis=1) * SCALE
    score = tl.where(valid_token, score, -float("inf"))
    probability = tl.where(valid_token, tl.exp(score - tl.max(score, axis=0)), 0.0)
    denominator = tl.sum(probability, axis=0)
    v = tl.load(
        value_cache_ptr + cache_base[:, None] + dim_offsets[None, :],
        mask=cache_mask, other=0.0,
    ).to(tl.float32)
    numerator = tl.sum(probability[:, None] * v, axis=0)
    result = tl.where(length > 0, numerator / denominator, 0.0)
    tl.store(out_ptr + batch_head * D + dim_offsets, result, mask=dim_offsets < D)


def op073_paged_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    max_sequence: int,
) -> torch.Tensor:
    """Attend from ``q[B,H,D]`` through a paged ``[P,S,H,D]`` KV cache."""
    _require_cuda_contiguous(q, key_cache, value_cache, page_table, lengths)
    _require_floating(q, key_cache, value_cache)
    if q.ndim != 3 or key_cache.ndim != 4 or page_table.ndim != 2:
        raise ValueError("expected q[B,H,D], cache[P,S,H,D], and table[B,max_pages]")
    batch, heads, dim = q.shape
    page_size = key_cache.shape[1]
    max_pages = page_table.shape[1]
    if key_cache.shape != value_cache.shape or key_cache.shape[2:] != (heads, dim):
        raise ValueError("key/value cache shapes are incompatible with q")
    if page_table.shape[0] != batch or lengths.shape != (batch,):
        raise ValueError("page table or lengths shape mismatch")
    if page_table.dtype != torch.int32 or lengths.dtype != torch.int32:
        raise TypeError("page_table and lengths must be int32")
    if heads < 1 or page_size < 1 or max_pages < 1 or dim < 1:
        raise ValueError("page size, page count, and dim must be positive")
    if dim > 256:
        raise ValueError("this teaching kernel requires dim <= 256")
    if max_sequence < 1 or max_sequence > min(1_024, page_size * max_pages):
        raise ValueError("max_sequence exceeds the table capacity or teaching limit 1024")
    # Metadata validation is intentionally synchronous in this safe teaching wrapper.
    host_lengths = lengths.detach().cpu()
    host_table = page_table.detach().cpu()
    if bool(((host_lengths < 0) | (host_lengths > max_sequence)).any()):
        raise ValueError("lengths must be in [0, max_sequence]")
    pages_used = (host_lengths + page_size - 1) // page_size
    for batch_index in range(batch):
        used = int(pages_used[batch_index])
        if used and bool(((host_table[batch_index, :used] < 0) |
                          (host_table[batch_index, :used] >= key_cache.shape[0])).any()):
            raise ValueError("page_table contains an invalid physical page id")
    out = torch.empty_like(q, dtype=torch.float32)
    if out.numel():
        _op073_paged_attention_kernel[(batch * heads,)](
            q, key_cache, value_cache, page_table, lengths, out, HEADS=heads, D=dim,
            PAGE_SIZE=page_size, MAX_PAGES=max_pages, MAX_SEQUENCE=max_sequence,
            SCALE=dim ** -0.5, BLOCK_T=triton.next_power_of_2(max_sequence),
            BLOCK_D=triton.next_power_of_2(dim), num_warps=8,
        )
    return out
