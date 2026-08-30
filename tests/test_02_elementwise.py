"""PyTorch-reference tests for elementwise operators 012--033."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from _utils import assert_close, load_category, require_gpu

require_gpu(pytest, torch)
ops = load_category("02_elementwise")


@pytest.mark.parametrize("n", [1, 31, 256, 257, 1003])
def test_binary_and_tail_masks(n):
    x = torch.randn(n, device="cuda")
    y = torch.randn_like(x).add_(0.5)
    for function, expected in (
        (ops.op012_sub, x - y),
        (ops.op013_mul, x * y),
        (ops.op014_div, x / y),
    ):
        assert_close(torch, function(x, y), expected)


def test_scalar_and_axpby():
    x = torch.randn(1003, device="cuda")
    y = torch.randn_like(x)
    assert_close(torch, ops.op015_scalar_add(x, 1.25), x + 1.25)
    assert_close(torch, ops.op016_axpby(x, y, 2.0, -0.5), 2.0 * x - 0.5 * y)


@pytest.mark.parametrize("name,reference,kwargs", [
    ("op017_relu", F.relu, {}),
    ("op018_leaky_relu", F.leaky_relu, {"negative_slope": 0.2}),
    ("op019_sigmoid", torch.sigmoid, {}),
    ("op020_tanh", torch.tanh, {}),
    ("op021_gelu", F.gelu, {"approximate": "tanh"}),
    ("op022_silu", F.silu, {}),
    ("op023_softplus", F.softplus, {}),
    ("op024_elu", F.elu, {"alpha": 0.7}),
    ("op025_hard_sigmoid", F.hardsigmoid, {}),
    ("op026_hard_swish", F.hardswish, {}),
    ("op027_square", torch.square, {}),
    ("op028_sqrt", torch.sqrt, {}),
    ("op029_exp", torch.exp, {}),
    ("op030_log", torch.log, {}),
])
def test_unary(name, reference, kwargs):
    x = torch.linspace(-5, 5, 1003, device="cuda")
    if name in ("op028_sqrt", "op030_log"):
        x = x.abs() + 0.01
    call = {}
    if name == "op018_leaky_relu": call["slope"] = kwargs["negative_slope"]
    if name == "op024_elu": call["alpha"] = kwargs["alpha"]
    assert_close(torch, getattr(ops, name)(x, **call), reference(x, **kwargs))


def test_select_clamp_and_row_broadcast():
    x = torch.randn(7, 257, device="cuda")
    y = torch.randn_like(x)
    bias = torch.randn(257, device="cuda")
    assert_close(torch, ops.op031_clamp(x, -0.5, 0.8), x.clamp(-0.5, 0.8))
    assert_close(torch, ops.op032_where(x > 0, x, y), torch.where(x > 0, x, y))
    assert_close(torch, ops.op033_row_bias_add(x, bias), x + bias)


def test_extreme_numerics():
    x = torch.tensor([-1000.0, -100.0, 0.0, 100.0, 1000.0], device="cuda")
    assert_close(torch, ops.op019_sigmoid(x), torch.sigmoid(x))
    assert_close(torch, ops.op023_softplus(x), F.softplus(x))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dtype_empty_and_boundary_contracts(dtype):
    x = torch.linspace(-3, 3, 257, device="cuda", dtype=dtype)
    y = torch.linspace(2, 4, 257, device="cuda", dtype=dtype)
    assert_close(torch, ops.op012_sub(x, y), x - y)
    assert_close(torch, ops.op016_axpby(x, y, 0.5, -2.0), 0.5 * x - 2.0 * y)
    assert_close(torch, ops.op026_hard_swish(x), F.hardswish(x))
    assert_close(torch, ops.op029_exp(x), torch.exp(x))

    empty = torch.empty(0, device="cuda", dtype=dtype)
    assert ops.op012_sub(empty, empty).shape == empty.shape
    assert ops.op023_softplus(empty).shape == empty.shape
    matrix = torch.empty((0, 7), device="cuda", dtype=dtype)
    bias = torch.ones(7, device="cuda", dtype=dtype)
    assert ops.op033_row_bias_add(matrix, bias).shape == matrix.shape


def test_elementwise_invalid_contracts():
    x = torch.ones(2, 3, device="cuda")
    with pytest.raises(ValueError):
        ops.op012_sub(x, torch.ones(3, 2, device="cuda"))
    with pytest.raises(ValueError):
        ops.op031_clamp(x, 1.0, -1.0)
    with pytest.raises(TypeError):
        ops.op032_where(torch.ones_like(x), x, x)
    with pytest.raises(ValueError):
        ops.op033_row_bias_add(x, torch.ones(2, device="cuda"))
