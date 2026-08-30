"""Reference tests for losses and optimizers 089--100."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from _utils import assert_close, load_category, require_gpu

require_gpu(pytest, torch)
ops = load_category("09_losses_optimizers")
D = "cuda"
LOSS_CASES = [
    pytest.param(89, id="op089-acceptance"),
    pytest.param(90, id="op090-acceptance"),
    pytest.param(91, id="op091-acceptance"),
    pytest.param(92, id="op092-acceptance"),
    pytest.param(93, id="op093-acceptance"),
]
OPTIMIZER_CASES = [
    pytest.param(94, id="op094-acceptance"),
    pytest.param(95, id="op095-acceptance"),
    pytest.param(96, id="op096-acceptance"),
    pytest.param(97, id="op097-acceptance"),
    pytest.param(98, id="op098-acceptance"),
    pytest.param(99, id="op099-acceptance"),
    pytest.param(100, id="op100-acceptance"),
]


def test_op089_to_093_losses():
    a = torch.randn(7, 31, device=D); b = torch.randn_like(a)
    assert_close(torch, ops.op089_mse(a, b), (a - b).square().mean(-1))
    targets = torch.rand_like(a)
    assert_close(torch, ops.op090_bce_with_logits(a, targets), F.binary_cross_entropy_with_logits(a, targets, reduction="none"))
    labels = torch.randint(0, 31, (7,), device=D)
    assert_close(torch, ops.op091_cross_entropy(a, labels), F.cross_entropy(a, labels, reduction="none"))
    assert_close(torch, ops.op092_cosine_similarity(a, b), F.cosine_similarity(a, b, dim=-1))
    log_p, log_q = a.log_softmax(-1), b.log_softmax(-1)
    assert_close(torch, ops.op093_kl_divergence(log_p, log_q), (log_p.exp() * (log_p - log_q)).sum(-1))


def test_op092_matches_pytorch_epsilon_semantics():
    eps = 1e-4
    # Both nonzero norms are below eps.  PyTorch clamps each norm separately,
    # so this distinguishes its denominator from max(norm_a * norm_b, eps).
    a = torch.tensor([[eps / 2, 0.0], [0.0, 0.0]], device=D)
    b = torch.tensor([[eps / 2, 0.0], [1.0, -2.0]], device=D)
    assert_close(
        torch,
        ops.op092_cosine_similarity(a, b, eps),
        F.cosine_similarity(a, b, dim=-1, eps=eps),
    )


def test_op093_zero_probability_extended_real_boundaries():
    negative_infinity = -float("inf")
    log_p = torch.tensor(
        [[negative_infinity, 0.0], [negative_infinity, 0.0], [0.0, negative_infinity]],
        device=D,
    )
    log_q = torch.tensor(
        [[negative_infinity, 0.0], [-2.0, 0.0], [negative_infinity, 0.0]],
        device=D,
    )
    # P=0 contributes 0 for either Q=0 or Q>0; P>0,Q=0 contributes +inf.
    expected = torch.tensor([0.0, 0.0, float("inf")], device=D)
    assert_close(torch, ops.op093_kl_divergence(log_p, log_q), expected)


def test_op094_sgd_and_095_momentum():
    p = torch.randn(1003, device=D); g = torch.randn_like(p)
    assert_close(torch, ops.op094_sgd_(p.clone(), g, 0.03), p - 0.03 * g)
    velocity = torch.randn_like(p); expected_v = 0.9 * velocity + g
    actual_p = p.clone(); actual_v = velocity.clone()
    ops.op095_momentum_sgd_(actual_p, g, actual_v, 0.03, 0.9)
    assert_close(torch, actual_v, expected_v); assert_close(torch, actual_p, p - 0.03 * expected_v)


def test_op096_adam_and_097_adamw():
    p = torch.randn(1003, device=D); g = torch.randn_like(p)
    m = torch.randn_like(p); v = torch.rand_like(p)
    for adamw in (False, True):
        p0, m0, v0 = p.clone(), m.clone(), v.clone()
        beta1, beta2, step, lr, eps = 0.9, 0.999, 3, 1e-3, 1e-8
        mt = beta1 * m0 + (1 - beta1) * g
        vt = beta2 * v0 + (1 - beta2) * g.square()
        update = (mt / (1 - beta1 ** step)) / (torch.sqrt(vt / (1 - beta2 ** step)) + eps)
        if adamw: update += 0.01 * p0
        expected = p0 - lr * update
        if adamw: ops.op097_adamw_(p0, g, m0, v0, step, lr, beta1, beta2, eps, 0.01)
        else: ops.op096_adam_(p0, g, m0, v0, step, lr, beta1, beta2, eps)
        assert_close(torch, p0, expected); assert_close(torch, m0, mt); assert_close(torch, v0, vt)


def test_op098_to_100_training_utilities():
    p = torch.randn(1003, device=D); g = torch.randn_like(p); state = torch.rand_like(p)
    expected_state = state + g.square(); expected_p = p - 0.02 * g / (expected_state.sqrt() + 1e-10)
    actual_p, actual_state = p.clone(), state.clone()
    ops.op098_adagrad_(actual_p, g, actual_state, 0.02)
    assert_close(torch, actual_p, expected_p); assert_close(torch, actual_state, expected_state)
    norm = g.norm().item(); expected_g = g * min(1.0, 0.5 / (norm + 1e-6))
    assert_close(torch, ops.op099_gradient_norm_clip_(g.clone(), norm, 0.5), expected_g)
    assert ops.op100_non_finite_check(torch.tensor([1.0, float("inf")], device=D)).item() == 1
    assert ops.op100_non_finite_check(torch.tensor([1.0, 2.0], device=D)).item() == 0


@pytest.mark.parametrize("number", LOSS_CASES)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
def test_loss_acceptance_dtype_tail_and_zero_rows(number, dtype):
    rows, cols = 3, 257
    a = torch.linspace(-8.0, 8.0, rows * cols, device=D, dtype=dtype).reshape(rows, cols)
    b = torch.flip(a, dims=(-1,)).contiguous()
    if number == 89:
        actual, expected = ops.op089_mse(a, b), (a.float() - b.float()).square().mean(-1)
    elif number == 90:
        target = torch.linspace(0.0, 1.0, rows * cols, device=D, dtype=dtype).reshape_as(a)
        actual = ops.op090_bce_with_logits(a, target)
        expected = F.binary_cross_entropy_with_logits(a.float(), target.float(), reduction="none")
    elif number == 91:
        labels = torch.tensor([0, 128, 256], device=D, dtype=torch.int64)
        actual, expected = ops.op091_cross_entropy(a, labels), F.cross_entropy(a.float(), labels, reduction="none")
    elif number == 92:
        actual, expected = ops.op092_cosine_similarity(a, b), F.cosine_similarity(a.float(), b.float(), dim=-1)
    else:
        log_p, log_q = a.float().log_softmax(-1).to(dtype), b.float().log_softmax(-1).to(dtype)
        actual = ops.op093_kl_divergence(log_p, log_q)
        expected = (log_p.float().exp() * (log_p.float() - log_q.float())).sum(-1)
    assert_close(torch, actual, expected)

    empty = torch.empty((0, cols), device=D, dtype=dtype)
    if number == 89:
        output = ops.op089_mse(empty, empty)
    elif number == 90:
        output = ops.op090_bce_with_logits(empty, empty)
    elif number == 91:
        output = ops.op091_cross_entropy(empty, torch.empty(0, device=D, dtype=torch.int64))
    elif number == 92:
        output = ops.op092_cosine_similarity(empty, empty)
    else:
        output = ops.op093_kl_divergence(empty, empty)
    assert output.numel() == 0


def test_op090_acceptance_extreme_logits_are_stable():
    logits = torch.tensor([-1000.0, 1000.0, -80.0, 80.0], device=D)
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0], device=D)
    actual = ops.op090_bce_with_logits(logits, targets)
    expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert torch.isfinite(actual).all()
    assert_close(torch, actual, expected)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("shape", id="op089-invalid-shape"),
        pytest.param("label_dtype", id="op091-invalid-label-dtype"),
        pytest.param("label_range", id="op091-invalid-label-range"),
        pytest.param("eps", id="op092-invalid-eps"),
        pytest.param("kl_shape", id="op093-invalid-shape"),
    ],
)
def test_loss_invalid_contract_matrix(case):
    a = torch.randn(3, 17, device=D)
    b = torch.randn_like(a)
    if case == "shape":
        with pytest.raises(ValueError):
            ops.op089_mse(a, b[:, :-1].contiguous())
    elif case == "label_dtype":
        with pytest.raises(TypeError):
            ops.op091_cross_entropy(a, torch.zeros(3, device=D))
    elif case == "label_range":
        with pytest.raises(ValueError):
            ops.op091_cross_entropy(a, torch.tensor([0, 17, 1], device=D))
    elif case == "eps":
        for eps in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                ops.op092_cosine_similarity(a, b, eps)
    else:
        with pytest.raises(ValueError):
            ops.op093_kl_divergence(a, b[:, :-1].contiguous())


@pytest.mark.parametrize("number", OPTIMIZER_CASES)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
def test_optimizer_acceptance_zero_elements_and_tail(number, dtype):
    count = 1003
    p = torch.linspace(-2.0, 2.0, count, device=D, dtype=dtype)
    g = torch.linspace(1.0, -1.0, count, device=D, dtype=dtype)
    if number == 94:
        assert_close(torch, ops.op094_sgd_(p.clone(), g, 0.0), p)
        empty = ops.op094_sgd_(p[:0].clone(), g[:0], 0.1)
    elif number == 95:
        velocity = torch.zeros_like(p)
        assert_close(torch, ops.op095_momentum_sgd_(p.clone(), g, velocity.clone(), 0.0), p)
        empty = ops.op095_momentum_sgd_(p[:0].clone(), g[:0], velocity[:0].clone(), 0.1)
    elif number == 96:
        state = torch.zeros_like(p)
        assert_close(torch, ops.op096_adam_(p.clone(), g, state.clone(), state.clone(), 1, 0.0), p)
        empty = ops.op096_adam_(p[:0].clone(), g[:0], state[:0].clone(), state[:0].clone(), 1)
    elif number == 97:
        state = torch.zeros_like(p)
        assert_close(torch, ops.op097_adamw_(p.clone(), g, state.clone(), state.clone(), 1, 0.0), p)
        empty = ops.op097_adamw_(p[:0].clone(), g[:0], state[:0].clone(), state[:0].clone(), 1)
    elif number == 98:
        state = torch.zeros_like(p)
        assert_close(torch, ops.op098_adagrad_(p.clone(), g, state.clone(), 0.0), p)
        empty = ops.op098_adagrad_(p[:0].clone(), g[:0], state[:0].clone())
    elif number == 99:
        assert_close(torch, ops.op099_gradient_norm_clip_(g.clone(), g.norm().item(), 0.0), torch.zeros_like(g))
        empty = ops.op099_gradient_norm_clip_(g[:0].clone(), 0.0, 1.0)
    else:
        bad = torch.cat((p, torch.tensor([float("nan")], device=D, dtype=dtype)))
        assert ops.op100_non_finite_check(bad).item() == 1
        empty = ops.op100_non_finite_check(p[:0])
    assert empty.numel() in (0, 1)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("sgd_lr", id="op094-invalid-lr"),
        pytest.param("momentum", id="op095-invalid-momentum"),
        pytest.param("adam", id="op096-invalid-hyperparameters"),
        pytest.param("adamw", id="op097-invalid-hyperparameters"),
        pytest.param("adagrad", id="op098-invalid-eps"),
        pytest.param("clip", id="op099-invalid-scalars"),
        pytest.param("check_dtype", id="op100-invalid-dtype"),
    ],
)
def test_optimizer_invalid_contract_matrix(case):
    p = torch.ones(17, device=D)
    g = torch.ones_like(p)
    state = torch.zeros_like(p)
    if case == "sgd_lr":
        for value in (-1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                ops.op094_sgd_(p.clone(), g, value)
    elif case == "momentum":
        for value in (-1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                ops.op095_momentum_sgd_(p.clone(), g, state.clone(), 0.1, value)
    elif case == "adam":
        for kwargs in ({"step": 0}, {"step": 1, "beta1": float("nan")}, {"step": 1, "eps": float("inf")}):
            with pytest.raises(ValueError):
                ops.op096_adam_(p.clone(), g, state.clone(), state.clone(), **kwargs)
    elif case == "adamw":
        for decay in (-1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                ops.op097_adamw_(p.clone(), g, state.clone(), state.clone(), 1, weight_decay=decay)
    elif case == "adagrad":
        for eps in (0.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                ops.op098_adagrad_(p.clone(), g, state.clone(), eps=eps)
    elif case == "clip":
        for scalars in ((float("nan"), 1.0, 1e-6), (1.0, float("inf"), 1e-6), (1.0, 1.0, 0.0)):
            with pytest.raises(ValueError):
                ops.op099_gradient_norm_clip_(g.clone(), *scalars)
    else:
        with pytest.raises(TypeError):
            ops.op100_non_finite_check(torch.ones(17, device=D, dtype=torch.int32))
