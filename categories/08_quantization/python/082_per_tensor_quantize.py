"""082: symmetric per-tensor INT8 quantization."""

import torch
import triton
import triton.language as tl

from ._common import (
    _require_cuda_contiguous,
    _require_floating,
    _require_scalar_scale,
    _op082_round_and_clamp_int8,
)


@triton.jit
def _op082_per_tensor_quantize_kernel(
    x_ptr, scale_ptr, q_ptr, count, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    value = tl.load(x_ptr + offset, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    tl.store(q_ptr + offset, _op082_round_and_clamp_int8(value / scale), mask=mask)


def op082_per_tensor_quantize(
    x: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Quantize ``x`` using one caller-provided positive scalar scale."""
    _require_cuda_contiguous(x, scale)
    _require_floating(x, scale)
    _require_scalar_scale(scale)
    q = torch.empty_like(x, dtype=torch.int8)
    count = x.numel()
    if count:
        _op082_per_tensor_quantize_kernel[(triton.cdiv(count, 256),)](
            x, scale, q, count, BLOCK=256, num_warps=4
        )
    return q
