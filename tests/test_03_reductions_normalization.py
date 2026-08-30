"""Reference tests for reductions/normalization 034--046."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from _utils import assert_close, load_category, require_gpu

require_gpu(pytest, torch)
ops = load_category("03_reductions_normalization")
D = "cuda"
ROW_CASES = [
    pytest.param("op034_row_sum", lambda x: x.float().sum(-1), id="op034-acceptance"),
    pytest.param("op035_row_mean", lambda x: x.float().mean(-1), id="op035-acceptance"),
    pytest.param("op036_row_max", lambda x: x.max(-1).values, id="op036-acceptance"),
    pytest.param("op037_row_min", lambda x: x.min(-1).values, id="op037-acceptance"),
    pytest.param("op038_l1_norm", lambda x: x.float().abs().sum(-1), id="op038-acceptance"),
    pytest.param("op039_l2_norm", lambda x: torch.linalg.vector_norm(x.float(), dim=-1), id="op039-acceptance"),
    pytest.param("op040_variance", lambda x: x.float().var(-1, correction=0), id="op040-acceptance"),
    pytest.param("op041_argmax", lambda x: x.argmax(-1), id="op041-acceptance"),
    pytest.param("op042_log_softmax", lambda x: torch.log_softmax(x.float(), dim=-1).to(x.dtype), id="op042-acceptance"),
    pytest.param("op046_cumsum", lambda x: x.float().cumsum(-1).to(x.dtype), id="op046-acceptance"),
]


def test_basic_row_reductions():
    x = torch.randn(11, 257, device="cuda")
    references = {
        "op034_row_sum": x.float().sum(-1),
        "op035_row_mean": x.float().mean(-1),
        "op036_row_max": x.max(-1).values,
        "op037_row_min": x.min(-1).values,
        "op038_l1_norm": x.float().abs().sum(-1),
        "op039_l2_norm": torch.linalg.vector_norm(x.float(), dim=-1),
        "op040_variance": x.float().var(-1, correction=0),
        "op041_argmax": x.argmax(-1),
        "op042_log_softmax": torch.log_softmax(x.float(), dim=-1),
        "op046_cumsum": x.cumsum(-1),
    }
    for name, expected in references.items():
        assert_close(torch, getattr(ops, name)(x), expected)


def test_rms_norm():
    x = torch.randn(13, 193, device="cuda")
    gamma = torch.randn(193, device="cuda")
    expected = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5) * gamma
    assert_close(torch, ops.op043_rms_norm(x, gamma), expected)


def test_batch_norm_inference():
    x = torch.randn(23, 64, device="cuda")
    mean = torch.randn(64, device="cuda")
    var = torch.rand(64, device="cuda") + 0.2
    gamma = torch.randn(64, device="cuda")
    beta = torch.randn(64, device="cuda")
    expected = (x - mean) * torch.rsqrt(var + 1e-5) * gamma + beta
    assert_close(torch, ops.op044_batch_norm_inference(x, mean, var, gamma, beta), expected)


def test_group_norm():
    x = torch.randn(4, 8, 17, device="cuda")
    gamma = torch.randn(8, device="cuda")
    beta = torch.randn(8, device="cuda")
    expected = F.group_norm(x, 4, gamma, beta, 1e-5)
    assert_close(torch, ops.op045_group_norm(x, gamma, beta, 4), expected)


def test_reduction_edge_values():
    x = torch.tensor([[3.0], [2.0]], device="cuda")
    assert_close(torch, ops.op034_row_sum(x), x[:, 0])
    assert_close(torch, ops.op040_variance(x), torch.zeros(2, device="cuda"))
    assert torch.equal(ops.op041_argmax(x), torch.zeros(2, device="cuda", dtype=torch.int64))


@pytest.mark.parametrize("name,reference", ROW_CASES)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
def test_row_operator_dtype_tail_and_single_column(name, reference, dtype):
    for cols in (1, 257):
        x = torch.linspace(-3.0, 5.0, 3 * cols, device=D, dtype=dtype).reshape(3, cols)
        assert_close(torch, getattr(ops, name)(x), reference(x))


@pytest.mark.parametrize("name,reference", ROW_CASES)
def test_row_operator_zero_rows(name, reference):
    x = torch.empty((0, 17), device=D)
    actual = getattr(ops, name)(x)
    expected = reference(x)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype


def test_op041_acceptance_leftmost_tie_and_extreme_values():
    x = torch.tensor(
        [[-float("inf"), 7.0, 7.0, 1.0], [-9.0e20, -8.0e20, -8.0e20, -9.0e20]],
        device=D,
    )
    assert torch.equal(ops.op041_argmax(x), x.argmax(-1))


@pytest.mark.parametrize("name,_reference", ROW_CASES)
def test_row_operator_invalid_contract(name, _reference):
    function = getattr(ops, name)
    with pytest.raises(ValueError):
        function(torch.empty((3, 0), device=D))
    with pytest.raises(ValueError):
        function(torch.empty((3, 4, 5), device=D))
    with pytest.raises(TypeError):
        function(torch.ones((3, 5), device=D, dtype=torch.int32))


@pytest.mark.parametrize("eps", [-1.0, float("nan"), float("inf")])
def test_normalization_rejects_invalid_eps(eps):
    x = torch.ones((2, 8), device=D)
    gamma = torch.ones(8, device=D)
    beta = torch.zeros(8, device=D)
    variance = torch.ones(8, device=D)
    with pytest.raises(ValueError):
        ops.op043_rms_norm(x, gamma, eps)
    with pytest.raises(ValueError):
        ops.op044_batch_norm_inference(x, beta, variance, gamma, beta, eps)
    with pytest.raises(ValueError):
        ops.op045_group_norm(x.reshape(2, 8, 1), gamma, beta, 4, eps)


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
def test_op043_to_op045_acceptance_dtype_constant_and_zero_batch(dtype):
    x = torch.full((3, 8), 2.0, device=D, dtype=dtype)
    gamma = torch.linspace(0.5, 1.5, 8, device=D, dtype=dtype)
    beta = torch.linspace(-0.5, 0.5, 8, device=D, dtype=dtype)
    rms_expected = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        * gamma.float()
    ).to(dtype)
    assert_close(torch, ops.op043_rms_norm(x, gamma), rms_expected)
    bn_expected = (
        (x.float() - beta.float())
        * torch.rsqrt(torch.ones_like(beta, dtype=torch.float32) + 1e-5)
        * gamma.float()
        + beta.float()
    ).to(dtype)
    assert_close(torch, ops.op044_batch_norm_inference(x, beta, torch.ones_like(beta), gamma, beta), bn_expected)
    group_x = x.reshape(3, 8, 1)
    assert_close(torch, ops.op045_group_norm(group_x, gamma, beta, 4), F.group_norm(group_x, 4, gamma, beta, 1e-5))

    empty = torch.empty((0, 8), device=D, dtype=dtype)
    assert ops.op043_rms_norm(empty, gamma).shape == empty.shape
    assert ops.op044_batch_norm_inference(empty, beta, torch.ones_like(beta), gamma, beta).shape == empty.shape
    assert ops.op045_group_norm(empty.reshape(0, 8, 1), gamma, beta, 4).shape == (0, 8, 1)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("gamma_shape", id="op043-invalid-gamma"),
        pytest.param("bn_dtype", id="op044-invalid-dtype"),
        pytest.param("groups", id="op045-invalid-groups"),
    ],
)
def test_normalization_invalid_parameter_matrix(case):
    x = torch.ones((2, 8), device=D)
    gamma = torch.ones(8, device=D)
    beta = torch.zeros(8, device=D)
    if case == "gamma_shape":
        with pytest.raises(ValueError):
            ops.op043_rms_norm(x, gamma[:-1])
    elif case == "bn_dtype":
        with pytest.raises(TypeError):
            ops.op044_batch_norm_inference(x, beta, gamma, gamma.half(), beta)
    else:
        with pytest.raises(ValueError):
            ops.op045_group_norm(x.reshape(2, 8, 1), gamma, beta, 3)


def test_op046_multitile_scan_and_reduction_width_limit():
    x = torch.randn(3, 513, device=D)
    assert_close(torch, ops.op046_cumsum(x), x.float().cumsum(-1).to(x.dtype))
    too_wide = torch.empty((0, ops.MAX_REDUCTION_SIZE + 1), device=D)
    with pytest.raises(ValueError):
        ops.op034_row_sum(too_wide)
