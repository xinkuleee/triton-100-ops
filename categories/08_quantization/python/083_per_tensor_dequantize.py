"""083: symmetric per-tensor INT8 dequantization."""

import torch
import triton
import triton.language as tl

from ._common import _require_cuda_contiguous, _require_floating, _require_scalar_scale


@triton.jit
def _op083_per_tensor_dequantize_kernel(
    q_ptr, scale_ptr, out_ptr, count, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    quantized = tl.load(q_ptr + offset, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    tl.store(out_ptr + offset, quantized * scale, mask=mask)


def op083_per_tensor_dequantize(
    q: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize symmetric INT8 values to FP32."""
    _require_cuda_contiguous(q, scale)
    _require_floating(scale)
    if q.dtype != torch.int8:
        raise TypeError("q must have dtype int8")
    _require_scalar_scale(scale)
    out = torch.empty_like(q, dtype=torch.float32)
    count = q.numel()
    if count:
        _op083_per_tensor_dequantize_kernel[(triton.cdiv(count, 256),)](
            q, scale, out, count, BLOCK=256, num_warps=4
        )
    return out
