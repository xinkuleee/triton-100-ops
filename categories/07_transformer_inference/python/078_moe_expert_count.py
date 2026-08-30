"""078: count token assignments for mixture-of-experts routing."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import _require_cuda_contiguous


@triton.jit
def _op078_moe_expert_count_kernel(
    expert_ids_ptr, counts_ptr, tokens, experts, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < tokens
    expert = tl.load(expert_ids_ptr + offset, mask=mask, other=0)
    valid = mask & (expert >= 0) & (expert < experts)
    tl.atomic_add(counts_ptr + expert, 1, mask=valid)


def op078_moe_expert_count(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    _require_cuda_contiguous(expert_ids)
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert_ids must be int32 or int64")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    counts = torch.zeros(num_experts, device=expert_ids.device, dtype=torch.int32)
    tokens = expert_ids.numel()
    if tokens:
        _op078_moe_expert_count_kernel[(triton.cdiv(tokens, 256),)](
            expert_ids, counts, tokens, num_experts, BLOCK=256, num_warps=4
        )
    return counts
