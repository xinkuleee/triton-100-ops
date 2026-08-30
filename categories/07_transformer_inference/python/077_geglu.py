"""077: elementwise GEGLU activation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import _require_cuda_contiguous, _require_floating


@triton.jit
def _op077_geglu_kernel(gate_ptr, value_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n
    gate = tl.load(gate_ptr + offset, mask=mask).to(tl.float32)
    value = tl.load(value_ptr + offset, mask=mask).to(tl.float32)
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    gelu = 0.5 * gate * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0))
    tl.store(out_ptr + offset, gelu * value, mask=mask)


def op077_geglu(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    _require_cuda_contiguous(gate, value)
    _require_floating(gate, value)
    if gate.shape != value.shape:
        raise ValueError("gate and value must have the same shape")
    out = torch.empty_like(gate)
    count = gate.numel()
    if count:
        _op077_geglu_kernel[(triton.cdiv(count, 256),)](
            gate, value, out, count, BLOCK=256, num_warps=4
        )
    return out
