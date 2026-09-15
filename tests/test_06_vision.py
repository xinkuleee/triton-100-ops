"""Reference tests for vision operators 061--068."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from _utils import assert_close, load_category, require_gpu

require_gpu(pytest, torch)
ops = load_category("06_vision")
DEVICE = "cuda"


def test_op061_conv1d():
    x = torch.randn(2, 5, 31, device=DEVICE)
    w = torch.randn(7, 5, 3, device=DEVICE)
    b = torch.randn(7, device=DEVICE)
    assert_close(torch, ops.op061_conv1d(x, w, b, stride=2, padding=1), F.conv1d(x, w, b, stride=2, padding=1))


def test_op062_conv2d():
    x = torch.randn(2, 3, 15, 17, device=DEVICE)
    w = torch.randn(5, 3, 3, 3, device=DEVICE)
    b = torch.randn(5, device=DEVICE)
    assert_close(torch, ops.op062_conv2d(x, w, b, padding=1), F.conv2d(x, w, b, padding=1))


def test_op063_depthwise_conv2d():
    x = torch.randn(2, 4, 13, 15, device=DEVICE)
    w = torch.randn(4, 1, 3, 3, device=DEVICE)
    b = torch.randn(4, device=DEVICE)
    assert_close(torch, ops.op063_depthwise_conv2d(x, w, b, padding=1), F.conv2d(x, w, b, padding=1, groups=4))


def test_op064_pointwise_conv2d():
    x = torch.randn(2, 5, 11, 13, device=DEVICE)
    w = torch.randn(7, 5, device=DEVICE)
    b = torch.randn(7, device=DEVICE)
    expected = F.conv2d(x, w[:, :, None, None], b)
    assert_close(torch, ops.op064_pointwise_conv2d(x, w, b), expected)


def test_op065_and_op066_pooling():
    x = torch.randn(2, 3, 17, 19, device=DEVICE)
    assert_close(torch, ops.op065_max_pool2d(x, (3, 3), 2, 1), F.max_pool2d(x, 3, 2, 1))
    assert_close(torch, ops.op066_avg_pool2d(x, (3, 3), 2, 1), F.avg_pool2d(x, 3, 2, 1, count_include_pad=False))


def test_op067_and_op068_resize():
    x = torch.randn(2, 3, 7, 9, device=DEVICE)
    assert_close(torch, ops.op067_nearest_resize(x, 13, 15), F.interpolate(x, (13, 15), mode="nearest"))
    assert_close(torch, ops.op068_bilinear_resize(x, 13, 15), F.interpolate(x, (13, 15), mode="bilinear", align_corners=False))


@pytest.mark.parametrize(
    "input_shape,output_size",
    [
        ((1, 1, 50000, 1), (50001, 1)),
        ((1, 1, 1, 50000), (1, 50001)),
    ],
    ids=["height", "width"],
)
def test_op067_coordinate_product_exceeds_int32(input_shape, output_size):
    values = torch.arange(50000, device=DEVICE, dtype=torch.float32)
    x = values.reshape(input_shape)
    # Upsizing N to N+1 repeats the first value, then copies every remaining value.
    expected = torch.cat((values[:1], values)).reshape(1, 1, *output_size)
    actual = ops.op067_nearest_resize(x, *output_size)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_op065_negative_padding_identity_and_op066_valid_count():
    x = -torch.arange(1, 17, device=DEVICE, dtype=torch.float32).reshape(1, 1, 4, 4)
    assert_close(
        torch,
        ops.op065_max_pool2d(x, (3, 3), 1, 1),
        F.max_pool2d(x, 3, 1, 1),
    )
    assert_close(
        torch,
        ops.op066_avg_pool2d(x, (3, 3), 1, 1),
        F.avg_pool2d(x, 3, 1, 1, count_include_pad=False),
    )


def test_op068_single_pixel_and_invalid_output_extent():
    x = torch.tensor([[[[3.25]]]], device=DEVICE)
    expected = x.expand(1, 1, 7, 5)
    assert_close(torch, ops.op068_bilinear_resize(x, 7, 5), expected)
    with pytest.raises(ValueError):
        ops.op062_conv2d(
            torch.randn(1, 1, 2, 2, device=DEVICE),
            torch.randn(1, 1, 5, 5, device=DEVICE),
            torch.zeros(1, device=DEVICE),
        )
