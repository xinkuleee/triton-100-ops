"""Reference tests for Transformer inference operators 069--081."""

import math
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from _utils import assert_close, load_category, require_gpu

require_gpu(pytest, torch)
ops = load_category("07_transformer_inference")
D = "cuda"


def test_op069_causal_attention():
    q = torch.randn(2, 3, 17, 32, device=D)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert_close(torch, ops.op069_causal_attention(q, k, v), expected)


def test_op070_rope():
    x = torch.randn(11, 3, 16, device=D)
    theta = torch.randn(11, 8, device=D)
    cos, sin = theta.cos(), theta.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    expected = torch.empty_like(x)
    expected[..., 0::2] = even * cos[:, None] - odd * sin[:, None]
    expected[..., 1::2] = even * sin[:, None] + odd * cos[:, None]
    assert_close(torch, ops.op070_rope(x, cos, sin), expected)


def test_op071_alibi_bias():
    scores = torch.randn(2, 4, 7, 9, device=D)
    slopes = torch.randn(4, device=D)
    q = torch.arange(7, device=D)[:, None]
    k = torch.arange(9, device=D)[None, :]
    expected = scores + slopes[None, :, None, None] * (k - q)
    assert_close(torch, ops.op071_alibi_bias(scores, slopes), expected)


def test_op072_kv_cache_append():
    new_k = torch.randn(2, 3, 2, 8, device=D)
    new_v = torch.randn_like(new_k)
    positions = torch.tensor([[1, 4], [0, 5]], device=D)
    ck = torch.zeros(2, 3, 7, 8, device=D)
    cv = torch.zeros_like(ck)
    expected_k, expected_v = ck.clone(), cv.clone()
    for b in range(2):
        for t in range(2):
            expected_k[b, :, positions[b, t]] = new_k[b, :, t]
            expected_v[b, :, positions[b, t]] = new_v[b, :, t]
    got_k, got_v = ops.op072_kv_cache_append(new_k, new_v, positions, ck, cv)
    assert_close(torch, got_k, expected_k); assert_close(torch, got_v, expected_v)


def test_op073_paged_attention():
    batch, heads, dim, pages, page_size = 2, 2, 8, 4, 4
    q = torch.randn(batch, heads, dim, device=D)
    k = torch.randn(pages, page_size, heads, dim, device=D)
    v = torch.randn_like(k)
    table = torch.tensor([[0, 1], [2, 3]], device=D, dtype=torch.int32)
    lengths = torch.tensor([6, 7], device=D, dtype=torch.int32)
    expected = torch.empty_like(q, dtype=torch.float32)
    for b in range(batch):
        logical_k = torch.cat([k[table[b, p]] for p in range(2)], dim=0)[:lengths[b]]
        logical_v = torch.cat([v[table[b, p]] for p in range(2)], dim=0)[:lengths[b]]
        score = torch.einsum("hd,thd->ht", q[b], logical_k) / math.sqrt(dim)
        expected[b] = torch.einsum("ht,thd->hd", score.softmax(-1), logical_v)
    assert_close(torch, ops.op073_paged_attention(q, k, v, table, lengths, 8), expected)


def test_op074_and_075_residual_norms():
    x = torch.randn(9, 64, device=D); residual = torch.randn_like(x)
    gamma = torch.randn(64, device=D); beta = torch.randn(64, device=D)
    z = x + residual
    expected_ln = F.layer_norm(z, (64,), gamma, beta)
    expected_rms = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5) * gamma
    assert_close(torch, ops.op074_residual_layer_norm(x, residual, gamma, beta), expected_ln)
    assert_close(torch, ops.op075_residual_rms_norm(x, residual, gamma), expected_rms)


def test_op076_and_077_gated_mlp():
    gate = torch.randn(1003, device=D); value = torch.randn_like(gate)
    assert_close(torch, ops.op076_swiglu(gate, value), F.silu(gate) * value)
    assert_close(torch, ops.op077_geglu(gate, value), F.gelu(gate, approximate="tanh") * value)


def test_op078_to_081_decode_utilities():
    expert = torch.tensor([0, 2, 2, 1, 3, 2], device=D)
    assert torch.equal(ops.op078_moe_expert_count(expert, 4), torch.bincount(expert, minlength=4).int())
    logits = torch.randn(3, 17, device=D)
    assert_close(torch, ops.op079_temperature_scale(logits, 0.7), logits / 0.7)
    assert torch.equal(ops.op080_greedy_decode(logits), logits.argmax(-1))
    ids = torch.tensor([[1, 3, 5], [0, 2, 4], [6, 7, 8]], device=D)
    expected = logits.clone()
    for b in range(3):
        chosen = expected[b, ids[b]]
        expected[b, ids[b]] = torch.where(chosen > 0, chosen / 1.2, chosen * 1.2)
    actual = ops.op081_repetition_penalty_(logits.clone(), ids, 1.2)
    assert_close(torch, actual, expected)


def test_op073_empty_sequence_returns_zero():
    q = torch.randn(1, 2, 8, device=D)
    key = torch.randn(2, 4, 2, 8, device=D)
    value = torch.randn_like(key)
    table = torch.tensor([[0, 1]], device=D, dtype=torch.int32)
    lengths = torch.zeros(1, device=D, dtype=torch.int32)
    assert_close(
        torch,
        ops.op073_paged_attention(q, key, value, table, lengths, 8),
        torch.zeros_like(q, dtype=torch.float32),
    )


def test_op080_left_tie_and_transformer_invalid_contracts():
    logits = torch.tensor([[2.0, 3.0, 3.0, -1.0]], device=D)
    assert torch.equal(ops.op080_greedy_decode(logits), torch.tensor([1], device=D))

    with pytest.raises(ValueError):
        ops.op069_causal_attention(
            torch.randn(1, 1, 3, 8, device=D),
            torch.randn(1, 1, 2, 8, device=D),
            torch.randn(1, 1, 2, 8, device=D),
        )
    with pytest.raises(ValueError):
        ops.op074_residual_layer_norm(
            torch.randn(2, 8, device=D),
            torch.randn(2, 8, device=D),
            torch.ones(8, device=D),
            torch.zeros(8, device=D),
            float("nan"),
        )
    with pytest.raises(ValueError):
        ops.op081_repetition_penalty_(
            torch.randn(1, 8, device=D),
            torch.tensor([[8]], device=D),
            1.2,
        )
