"""087: fake quantization without materializing an INT8 tensor."""

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
def _op087_fake_quantize_kernel(
    x_ptr, scale_ptr, out_ptr, count, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    value = tl.load(x_ptr + offset, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    quantized = _op082_round_and_clamp_int8(value / scale)
    tl.store(out_ptr + offset, quantized.to(tl.float32) * scale, mask=mask)


def op087_fake_quantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return FP32 ``dequantize(quantize(x))`` without materializing INT8."""
    _require_cuda_contiguous(x, scale)
    _require_floating(x, scale)
    _require_scalar_scale(scale)
    out = torch.empty_like(x, dtype=torch.float32)
    count = x.numel()
    if count:
        _op087_fake_quantize_kernel[(triton.cdiv(count, 256),)](
            x, scale, out, count, BLOCK=256, num_warps=4
        )
    return out
