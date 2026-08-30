"""089: row-wise mean squared error."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import require_cuda_contiguous, require_floating, row_meta


@triton.jit
def _op089_mse_kernel(prediction_ptr, target_ptr, out_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < cols
    difference = tl.load(
        prediction_ptr + row * cols + col, mask=mask, other=0.0
    ).to(tl.float32)
    difference -= tl.load(
        target_ptr + row * cols + col, mask=mask, other=0.0
    ).to(tl.float32)
    tl.store(out_ptr + row, tl.sum(difference * difference, axis=0) / cols)


def op089_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return one mean squared error value per row of matching ``[R,C]`` inputs."""
    require_cuda_contiguous(prediction, target)
    require_floating(prediction, target)
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("prediction and target must have matching shape [rows, cols]")
    rows, cols = prediction.shape
    block, warps = row_meta(cols)
    out = torch.empty(rows, device=prediction.device, dtype=torch.float32)
    if rows:
        _op089_mse_kernel[(rows,)](
            prediction, target, out, cols, BLOCK=block, num_warps=warps
        )
    return out
