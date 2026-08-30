"""Reference tests for quantization operators 082--088."""

import pytest

torch = pytest.importorskip("torch")

from _utils import assert_close, load_category, require_gpu

require_gpu(pytest, torch)
ops = load_category("08_quantization")


def quantize_reference(x, scale):
    value = (x.float() / scale).clamp(-127, 127)
    rounded = torch.where(value >= 0, torch.floor(value + 0.5), torch.ceil(value - 0.5))
    return rounded.to(torch.int8)


def test_op082_083_087_tensor_quantization():
    x = torch.randn(1003, device="cuda") * 3
    scale = torch.tensor(0.05, device="cuda")
    q = quantize_reference(x, scale)
    assert torch.equal(ops.op082_per_tensor_quantize(x, scale), q)
    assert_close(torch, ops.op083_per_tensor_dequantize(q, scale), q.float() * scale)
    assert_close(torch, ops.op087_fake_quantize(x, scale), q.float() * scale)


def test_op084_per_row_quantize():
    x = torch.randn(7, 193, device="cuda")
    q, scale = ops.op084_per_row_quantize(x)
    expected_scale = x.abs().amax(-1) / 127
    expected_scale = expected_scale.clamp_min(1e-12)
    assert_close(torch, scale, expected_scale)
    assert torch.equal(q, quantize_reference(x, expected_scale[:, None]))


def test_op085_per_channel_quantize():
    x = torch.randn(31, 17, device="cuda")
    q, scale = ops.op085_per_channel_quantize(x)
    expected_scale = x.abs().amax(0).div(127).clamp_min(1e-12)
    assert_close(torch, scale, expected_scale)
    assert torch.equal(q, quantize_reference(x, expected_scale[None, :]))


def test_op086_int8_matmul():
    a = torch.randint(-8, 8, (35, 33), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (33, 29), device="cuda", dtype=torch.int8)
    sa = torch.tensor(0.1, device="cuda")
    sb = torch.tensor(0.2, device="cuda")
    expected = (a.int() @ b.int()).float() * sa * sb
    assert_close(torch, ops.op086_int8_matmul(a, b, sa, sb), expected)


def test_op088_blockwise_quantize():
    x = torch.randn(1003, device="cuda")
    q, scales = ops.op088_blockwise_quantize(x, 256)
    for block, scale in enumerate(scales):
        values = x[block * 256:(block + 1) * 256]
        assert_close(torch, scale, values.abs().max().div(127).clamp_min(1e-12))
        assert torch.equal(q[block * 256:(block + 1) * 256], quantize_reference(values, scale))


def test_op082_rounding_saturation_and_op083_minus_128():
    scale = torch.tensor(1.0, device="cuda")
    x = torch.tensor([-200.0, -2.5, -0.5, 0.5, 2.5, 200.0], device="cuda")
    expected = torch.tensor([-127, -3, -1, 1, 3, 127], device="cuda", dtype=torch.int8)
    assert torch.equal(ops.op082_per_tensor_quantize(x, scale), expected)
    q = torch.tensor([-128, 127], device="cuda", dtype=torch.int8)
    assert_close(torch, ops.op083_per_tensor_dequantize(q, scale), q.float())


def test_op084_zero_row_and_invalid_scale_contracts():
    x = torch.zeros(2, 17, device="cuda")
    q, scales = ops.op084_per_row_quantize(x)
    assert torch.equal(q, torch.zeros_like(q))
    assert_close(torch, scales, torch.full_like(scales, 1e-12))
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            ops.op082_per_tensor_quantize(x, torch.tensor(value, device="cuda"))
