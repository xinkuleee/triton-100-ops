"""Reference tests for linear algebra operators 057--060."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from _utils import assert_close, load_category, require_gpu

require_gpu(pytest, torch)
ops = load_category("05_linear_algebra")


def test_op057_batched_matmul():
    a = torch.randn(3, 35, 17, device="cuda", dtype=torch.float16)
    b = torch.randn(3, 17, 29, device="cuda", dtype=torch.float16)
    assert_close(torch, ops.op057_batched_matmul(a, b), torch.bmm(a, b).float(), dtype=a.dtype)


def test_op058_gemv():
    a = torch.randn(37, 193, device="cuda")
    x = torch.randn(193, device="cuda")
    assert_close(torch, ops.op058_gemv(a, x), a @ x)


def test_op059_outer_product():
    x = torch.randn(37, device="cuda")
    y = torch.randn(53, device="cuda")
    assert_close(torch, ops.op059_outer_product(x, y), torch.outer(x, y))


def test_op060_linear_bias():
    x = torch.randn(35, 19, device="cuda", dtype=torch.float16)
    weight = torch.randn(27, 19, device="cuda", dtype=torch.float16)
    bias = torch.randn(27, device="cuda", dtype=torch.float16)
    assert_close(torch, ops.op060_linear_bias(x, weight, bias), F.linear(x, weight, bias).float(), dtype=x.dtype)


def test_op057_and_060_zero_inner_dimension():
    a = torch.empty(2, 3, 0, device="cuda")
    b = torch.empty(2, 0, 5, device="cuda")
    assert_close(torch, ops.op057_batched_matmul(a, b), torch.bmm(a, b))

    x = torch.empty(3, 0, device="cuda")
    weight = torch.empty(5, 0, device="cuda")
    bias = torch.randn(5, device="cuda")
    expected = bias.expand(3, 5).float()
    assert_close(torch, ops.op060_linear_bias(x, weight, bias), expected)


def test_op057_to_060_reject_shape_and_dtype_mismatch():
    with pytest.raises(ValueError):
        ops.op057_batched_matmul(
            torch.randn(2, 3, 4, device="cuda"),
            torch.randn(2, 5, 6, device="cuda"),
        )
    with pytest.raises(ValueError):
        ops.op059_outer_product(
            torch.randn(3, device="cuda"),
            torch.randn(4, device="cuda", dtype=torch.float16),
        )
