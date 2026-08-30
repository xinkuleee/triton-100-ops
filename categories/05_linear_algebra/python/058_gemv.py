"""058: dense matrix-vector multiplication."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import MAX_GEMV_K, require_tensors


@triton.jit
def _op058_gemv_kernel(
    matrix_ptr, vector_ptr, out_ptr, rows, cols, BLOCK_K: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_K)
    mask = col < cols
    matrix = tl.load(matrix_ptr + row * cols + col, mask=mask, other=0.0)
    vector = tl.load(vector_ptr + col, mask=mask, other=0.0)
    tl.store(out_ptr + row, tl.sum(matrix.to(tl.float32) * vector, axis=0), mask=row < rows)


def op058_gemv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Compute a matrix-vector product with one Triton program per row."""
    require_tensors(matrix, vector)
    if matrix.ndim != 2 or vector.ndim != 1 or matrix.shape[1] != vector.numel():
        raise ValueError("expected matrix[M,K] and vector[K]")
    rows, cols = matrix.shape
    if cols < 1 or cols > MAX_GEMV_K:
        raise ValueError(f"K must be in [1, {MAX_GEMV_K}]")
    out = torch.empty(rows, device=matrix.device, dtype=torch.float32)
    if rows:
        block = triton.next_power_of_2(cols)
        _op058_gemv_kernel[(rows,)](
            matrix, vector, out, rows, cols, BLOCK_K=block,
            num_warps=4 if block <= 2048 else 8,
        )
    return out
