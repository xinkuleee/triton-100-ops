"""079: temperature scaling for logits."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ._common import _require_cuda_contiguous, _require_floating


@triton.jit
def _op079_temperature_scale_kernel(
    logits_ptr, out_ptr, temperature, n, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n
    tl.store(out_ptr + offset, tl.load(logits_ptr + offset, mask=mask) / temperature, mask=mask)


def op079_temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    _require_cuda_contiguous(logits)
    _require_floating(logits)
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    out = torch.empty_like(logits)
    count = logits.numel()
    if count:
        _op079_temperature_scale_kernel[(triton.cdiv(count, 256),)](
            logits, out, temperature, count, BLOCK=256, num_warps=4
        )
    return out
