"""056: embedding-bag sum."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import (
    MAX_BAG,
    require_device_tensor,
    require_float_tensor,
    require_same_device,
    row_launch_meta,
)


@triton.jit
def _op056_embedding_bag_sum_kernel(
    weight_ptr, indices_ptr, offsets_ptr, out_ptr, vocab, width,
    MAX_BAG_SIZE: tl.constexpr, BLOCK_WIDTH: tl.constexpr,
):
    bag = tl.program_id(0)
    col = tl.arange(0, BLOCK_WIDTH)
    start = tl.load(offsets_ptr + bag)
    end = tl.load(offsets_ptr + bag + 1)
    accumulator = tl.zeros((BLOCK_WIDTH,), tl.float32)
    for step in range(0, MAX_BAG_SIZE):
        position = start + step
        active = position < end
        index = tl.load(indices_ptr + position, mask=active, other=0)
        valid = active & (index >= 0) & (index < vocab)
        accumulator += tl.load(
            weight_ptr + index * width + col,
            mask=valid & (col < width), other=0.0,
        ).to(tl.float32)
    tl.store(out_ptr + bag * width + col, accumulator, mask=col < width)


def op056_embedding_bag_sum(
    weight: torch.Tensor, indices: torch.Tensor, offsets: torch.Tensor,
    max_bag_size: int,
) -> torch.Tensor:
    """Sum embedding rows in half-open bags ``offsets[b]:offsets[b+1]``."""
    require_float_tensor(weight, "weight")
    for tensor, name in ((indices, "indices"), (offsets, "offsets")):
        require_device_tensor(tensor, name)
        if tensor.dtype != torch.int32 or tensor.ndim != 1:
            raise ValueError(f"{name} must be a contiguous int32 vector")
    require_same_device(weight, indices, offsets)
    if weight.ndim != 2 or offsets.numel() < 1:
        raise ValueError("expected weight[V,D] and at least one offset")
    if max_bag_size < 1 or max_bag_size > MAX_BAG:
        raise ValueError(f"max_bag_size must be in [1, {MAX_BAG}]")
    host_offsets = offsets.detach().cpu()
    if host_offsets[0].item() != 0 or host_offsets[-1].item() != indices.numel():
        raise ValueError("offsets must start at 0 and end at indices length")
    if bool((host_offsets[1:] < host_offsets[:-1]).any()):
        raise ValueError("offsets must be monotonic")
    if host_offsets.numel() > 1 and int((host_offsets[1:] - host_offsets[:-1]).max()) > max_bag_size:
        raise ValueError("a bag exceeds max_bag_size")
    vocab, width = weight.shape
    block, warps = row_launch_meta(width)
    bags = offsets.numel() - 1
    out = torch.empty((bags, width), device=weight.device, dtype=torch.float32)
    if bags:
        _op056_embedding_bag_sum_kernel[(bags,)](
            weight, indices, offsets, out, vocab, width,
            MAX_BAG_SIZE=max_bag_size, BLOCK_WIDTH=block, num_warps=warps,
        )
    return out
