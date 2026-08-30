"""069: causal scaled dot-product attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import _require_cuda_contiguous, _require_floating


@triton.jit
def _op069_causal_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    LQ: tl.constexpr,
    LK: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    query_index = row % LQ
    batch_head = row // LQ
    key_offsets = tl.arange(0, BLOCK_K)
    dim_offsets = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr + (batch_head * LQ + query_index) * D + dim_offsets,
        mask=dim_offsets < D,
        other=0.0,
    ).to(tl.float32)
    matrix_mask = (key_offsets[:, None] < LK) & (dim_offsets[None, :] < D)
    k = tl.load(
        k_ptr + (batch_head * LK + key_offsets[:, None]) * D + dim_offsets[None, :],
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(k * q[None, :], axis=1) * SCALE
    causal_mask = (key_offsets < LK) & (key_offsets <= query_index)
    scores = tl.where(causal_mask, scores, -float("inf"))
    scores -= tl.max(scores, axis=0)
    probabilities = tl.where(causal_mask, tl.exp(scores), 0.0)
    probabilities /= tl.sum(probabilities, axis=0)
    v = tl.load(
        v_ptr + (batch_head * LK + key_offsets[:, None]) * D + dim_offsets[None, :],
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)
    result = tl.sum(probabilities[:, None] * v, axis=0)
    tl.store(
        out_ptr + (batch_head * LQ + query_index) * D + dim_offsets,
        result,
        mask=dim_offsets < D,
    )


def op069_causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Return causal ``softmax(Q K^T / sqrt(D)) V`` for ``[B,H,L,D]`` tensors."""
    _require_cuda_contiguous(q, k, v)
    _require_floating(q, k, v)
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, length, dim]")
    batch, heads, query_length, dim = q.shape
    key_length = k.shape[2]
    if k.shape != v.shape or k.shape[:2] != (batch, heads) or k.shape[3] != dim:
        raise ValueError("q, k, and v shapes are incompatible")
    if key_length < 1 or dim < 1 or key_length > 256 or dim > 256:
        raise ValueError("this teaching kernel requires key_length and dim in [1, 256]")
    if query_length > key_length:
        raise ValueError(
            "causal query_length must not exceed key_length; this teaching "
            "kernel uses zero-based query/key positions without an offset"
        )
    out = torch.empty_like(q, dtype=torch.float32)
    if out.numel() == 0:
        return out
    block_k, block_d = triton.next_power_of_2(key_length), triton.next_power_of_2(dim)
    _op069_causal_attention_kernel[(batch * heads * query_length,)](
        q, k, v, out, LQ=query_length, LK=key_length, D=dim,
        SCALE=dim ** -0.5, BLOCK_K=block_k, BLOCK_D=block_d, num_warps=8,
    )
    return out
