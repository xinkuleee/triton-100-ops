"""054: CSR sparse matrix-vector multiplication."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import (
    MAX_ROW,
    require_device_tensor,
    require_float_tensor,
    require_same_device,
    row_launch_meta,
)


@triton.jit
def _op054_csr_spmv_kernel(
    row_ptr, col_idx, values, vector, out, rows, vector_size,
    BLOCK_NNZ: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK_NNZ)
    start = tl.load(row_ptr + row)
    end = tl.load(row_ptr + row + 1)
    position = start + lane
    active = position < end
    col = tl.load(col_idx + position, mask=active, other=0)
    valid = active & (col >= 0) & (col < vector_size)
    value = tl.load(values + position, mask=active, other=0.0).to(tl.float32)
    x = tl.load(vector + col, mask=valid, other=0.0)
    tl.store(out + row, tl.sum(tl.where(valid, value * x, 0.0), axis=0), mask=row < rows)


def op054_csr_spmv(
    row_ptr: torch.Tensor, col_idx: torch.Tensor, values: torch.Tensor,
    vector: torch.Tensor, max_nnz_per_row: int,
) -> torch.Tensor:
    """Multiply a CSR matrix by a dense vector, one program per CSR row."""
    for tensor, name in ((row_ptr, "row_ptr"), (col_idx, "col_idx")):
        require_device_tensor(tensor, name)
        if tensor.dtype != torch.int32 or tensor.ndim != 1:
            raise ValueError(f"{name} must be a contiguous int32 vector")
    require_float_tensor(values, "values")
    require_float_tensor(vector, "vector")
    require_same_device(values, row_ptr, col_idx, vector)
    if values.ndim != 1 or vector.ndim != 1 or col_idx.numel() != values.numel():
        raise ValueError("col_idx/values must be equal vectors and vector must be 1-D")
    if row_ptr.numel() < 1:
        raise ValueError("row_ptr must contain at least its initial zero")
    if max_nnz_per_row < 1 or max_nnz_per_row > MAX_ROW:
        raise ValueError(f"max_nnz_per_row must be in [1, {MAX_ROW}]")
    # Validation synchronizes intentionally; use a trusted-CSR fast wrapper in production.
    host_ptr = row_ptr.detach().cpu()
    if host_ptr[0].item() != 0 or host_ptr[-1].item() != values.numel():
        raise ValueError("row_ptr must start at 0 and end at nnz")
    if bool((host_ptr[1:] < host_ptr[:-1]).any()):
        raise ValueError("row_ptr must be monotonic")
    if host_ptr.numel() > 1 and int((host_ptr[1:] - host_ptr[:-1]).max()) > max_nnz_per_row:
        raise ValueError("a CSR row exceeds max_nnz_per_row")
    rows = row_ptr.numel() - 1
    block, warps = row_launch_meta(max_nnz_per_row)
    out = torch.empty(rows, device=values.device, dtype=torch.float32)
    if rows:
        _op054_csr_spmv_kernel[(rows,)](
            row_ptr, col_idx, values, vector, out, rows, vector.numel(),
            BLOCK_NNZ=block, num_warps=warps,
        )
    return out
