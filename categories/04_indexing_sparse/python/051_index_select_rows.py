"""051: complete-row index selection with zero rows for invalid indices."""

import torch
import triton
import triton.language as tl

from ._common import require_device_tensor, require_float_tensor, require_same_device, row_launch_meta


@triton.jit
def _op051_index_select_rows_kernel(
    x_ptr, indices_ptr, out_ptr, rows, cols, BLOCK_WIDTH: tl.constexpr
):
    selected_row = tl.program_id(0)
    col = tl.arange(0, BLOCK_WIDTH)
    source_row = tl.load(indices_ptr + selected_row).to(tl.int64)
    valid_row = (source_row >= 0) & (source_row < rows)
    value = tl.load(
        x_ptr + source_row * cols + col,
        mask=valid_row & (col < cols),
        other=0.0,
    )
    tl.store(out_ptr + selected_row * cols + col, value, mask=col < cols)


def op051_index_select_rows(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select complete rows; invalid row numbers produce zero rows."""
    require_float_tensor(x, "x")
    require_device_tensor(indices, "indices")
    require_same_device(x, indices)
    if x.ndim != 2 or indices.ndim != 1 or indices.dtype != torch.int64:
        raise ValueError("expected x[R,C] and int64 indices[I]")
    rows, cols = x.shape
    block, warps = row_launch_meta(cols)
    out = torch.empty((indices.numel(), cols), device=x.device, dtype=x.dtype)
    if indices.numel():
        _op051_index_select_rows_kernel[(indices.numel(),)](
            x,
            indices,
            out,
            rows,
            cols,
            BLOCK_WIDTH=block,
            num_warps=warps,
        )
    return out
