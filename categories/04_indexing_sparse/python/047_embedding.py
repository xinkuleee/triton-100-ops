"""047: embedding row lookup with zero rows for invalid indices."""

import torch
import triton
import triton.language as tl

from ._common import require_device_tensor, require_float_tensor, require_same_device, row_launch_meta


@triton.jit
def _op047_embedding_kernel(
    weight_ptr, indices_ptr, out_ptr, vocab, width, BLOCK_WIDTH: tl.constexpr
):
    item = tl.program_id(0)
    col = tl.arange(0, BLOCK_WIDTH)
    index = tl.load(indices_ptr + item).to(tl.int64)
    valid_index = (index >= 0) & (index < vocab)
    values = tl.load(
        weight_ptr + index * width + col,
        mask=valid_index & (col < width),
        other=0.0,
    )
    tl.store(out_ptr + item * width + col, values, mask=col < width)


def op047_embedding(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Lookup rows from ``weight[V,D]``; invalid indices yield zero rows."""
    require_float_tensor(weight, "weight")
    require_device_tensor(indices, "indices")
    require_same_device(weight, indices)
    if weight.ndim != 2 or indices.ndim != 1 or indices.dtype != torch.int64:
        raise ValueError("expected weight[V,D] and contiguous int64 indices[I]")
    vocab, width = weight.shape
    block, warps = row_launch_meta(width)
    out = torch.empty(
        (indices.numel(), width), device=weight.device, dtype=weight.dtype
    )
    if indices.numel():
        _op047_embedding_kernel[(indices.numel(),)](
            weight,
            indices,
            out,
            vocab,
            width,
            BLOCK_WIDTH=block,
            num_warps=warps,
        )
    return out
