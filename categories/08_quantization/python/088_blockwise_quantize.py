"""088: symmetric INT8 quantization with one scale per flat block."""

import torch
import triton
import triton.language as tl

from ._common import (
    MAX_REDUCTION_SIZE,
    _require_cuda_contiguous,
    _require_floating,
    _op082_round_and_clamp_int8,
)


@triton.jit
def _op088_blockwise_quantize_kernel(
    x_ptr, q_ptr, scales_ptr, count, BLOCK_SIZE: tl.constexpr
):
    block = tl.program_id(0)
    offset = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < count
    value = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    absolute_maximum = tl.max(tl.abs(value), axis=0)
    scale = tl.maximum(absolute_maximum / 127.0, 1.0e-12)
    tl.store(q_ptr + offset, _op082_round_and_clamp_int8(value / scale), mask=mask)
    tl.store(scales_ptr + block, scale)


def op088_blockwise_quantize(
    x: torch.Tensor, block_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize flattened consecutive blocks with one FP32 scale per block."""
    _require_cuda_contiguous(x)
    _require_floating(x)
    if (
        block_size <= 0
        or block_size > MAX_REDUCTION_SIZE
        or block_size & (block_size - 1)
    ):
        raise ValueError("block_size must be a power of two in [1, 65536]")
    count = x.numel()
    blocks = triton.cdiv(count, block_size)
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(blocks, device=x.device, dtype=torch.float32)
    if count:
        _op088_blockwise_quantize_kernel[(blocks,)](
            x, q, scales, count, BLOCK_SIZE=block_size,
            num_warps=4 if block_size <= 2_048 else 8,
        )
    return q, scales
