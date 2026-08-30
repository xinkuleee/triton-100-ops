"""Input factories for project-authored operators 012--100.

Each factory returns ``(triton_callable, reference_callable, bytes_moved)``.
``bytes_moved`` is deliberately ``None`` for reductions, GEMMs, attention, sparse
ops, and in-place optimizers where a single bandwidth number would be misleading.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import math
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parents[1]
RANGES = [
    (1, 11, "01_official_tutorials"),
    (12, 33, "02_elementwise"),
    (34, 46, "03_reductions_normalization"),
    (47, 56, "04_indexing_sparse"),
    (57, 60, "05_linear_algebra"),
    (61, 68, "06_vision"),
    (69, 81, "07_transformer_inference"),
    (82, 88, "08_quantization"),
    (89, 100, "09_losses_optimizers"),
]
SUPPORTED_CASES = set(range(12, 101))


_LOADED_MODULES = {}
_PACKAGES = {}


def _operator_path(number):
    directory = next(name for lo, hi, name in RANGES if lo <= number <= hi)
    matches = sorted(
        (ROOT / "categories" / directory / "python").glob(f"{number:03d}_*.py")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"operator {number:03d} must have exactly one numbered Python file; "
            f"found {len(matches)}"
        )
    return directory, matches[0]


def load_ops(number):
    """Load only the numbered implementation requested by the benchmark."""
    directory, path = _operator_path(number)
    cached = _LOADED_MODULES.get(path)
    if cached is not None:
        return cached

    package_name = f"_triton_100_ops_bench_{directory}"
    if package_name not in _PACKAGES:
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(path.parent)]
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name, loader=None, is_package=True
        )
        sys.modules[package_name] = package
        _PACKAGES[package_name] = package

    module_name = f"{package_name}.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load numbered implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    _LOADED_MODULES[path] = module
    return module


def _shape(size):
    return {"small": 2**14, "medium": 2**20, "large": 2**24}[size]


def _side(size):
    return {"small": 128, "medium": 512, "large": 1024}[size]


def _matrix_rows(size):
    return {"small": 16, "medium": 128, "large": 512}[size]


def _matrix_cols(size):
    return {"small": 64, "medium": 256, "large": 512}[size]


def _quantize_reference(x, scale):
    value = (x.float() / scale).clamp(-127, 127)
    rounded = torch.where(value >= 0, torch.floor(value + 0.5), torch.ceil(value - 0.5))
    return rounded.to(torch.int8)


def _none_case(tri, ref):
    return tri, ref, None


def _assert_outputs_close(actual, expected, path="output"):
    """Recursively compare tensor or nested tensor benchmark results."""
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        try:
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        except AssertionError as error:
            raise AssertionError(f"benchmark correctness check failed at {path}: {error}") from error
        return
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError(
                f"benchmark correctness check failed at {path}: "
                f"length {len(actual)} != {len(expected)}"
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_outputs_close(actual_item, expected_item, f"{path}[{index}]")
        return
    if actual != expected:
        raise AssertionError(
            f"benchmark correctness check failed at {path}: {actual!r} != {expected!r}"
        )


def validate_case(number, triton_callable, reference_callable):
    """Run one untimed correctness comparison."""
    with torch.no_grad():
        actual = triton_callable()
        expected = reference_callable()
    _assert_outputs_close(actual, expected)
    return None


def make_case(number, size="medium", device="cuda"):
    if number < 1 or number > 100:
        raise ValueError("operator number must be in [1, 100]")
    if number < 12:
        _, tutorial_path = _operator_path(number)
        raise RuntimeError(
            f"{number:03d} is a complete official tutorial script and is outside "
            f"the unified 012--100 benchmark runner. Run {tutorial_path} directly "
            "to preserve its original examples, checks, and any benchmark."
        )
    op = load_ops(number)
    n = _shape(size)
    # Most cases construct domain-specific inputs below.  Allocate the large
    # generic vectors only for operators that actually consume them; otherwise
    # e.g. an attention or convolution benchmark silently holds two unrelated
    # tensors for the entire run.
    x = y = None
    if 12 <= number <= 33:
        x = torch.randn(n, device=device)
        y = torch.randn_like(x).add_(0.5)

    if 12 <= number <= 33:
        return _make_elementwise_case(op, number, x, y)
    if 34 <= number <= 46:
        return _make_reduction_case(op, number, size, device)
    if 47 <= number <= 56:
        return _make_indexing_sparse_case(op, number, size, device)
    if 57 <= number <= 60:
        return _make_linear_algebra_case(op, number, size, device)
    if 61 <= number <= 68:
        return _make_vision_case(op, number, size, device)
    if 69 <= number <= 81:
        return _make_transformer_case(op, number, size, device)
    if 82 <= number <= 88:
        return _make_quantization_case(op, number, size, device)
    if 89 <= number <= 100:
        return _make_loss_optimizer_case(op, number, size, device)
    raise AssertionError(f"missing benchmark factory for operator {number:03d}")


def _make_elementwise_case(op, number, x, y):
    unary_x = x.abs() + 0.01 if number in (28, 30) else x
    mapping = {
        12: (lambda: op.op012_sub(x, y), lambda: x - y, 3 * x.nbytes),
        13: (lambda: op.op013_mul(x, y), lambda: x * y, 3 * x.nbytes),
        14: (lambda: op.op014_div(x, y), lambda: x / y, 3 * x.nbytes),
        15: (lambda: op.op015_scalar_add(x, 1.2), lambda: x + 1.2, 2 * x.nbytes),
        16: (lambda: op.op016_axpby(x, y, 2.0, -0.5), lambda: 2.0 * x - 0.5 * y, 3 * x.nbytes),
        17: (lambda: op.op017_relu(x), lambda: F.relu(x), 2 * x.nbytes),
        18: (lambda: op.op018_leaky_relu(x, 0.1), lambda: F.leaky_relu(x, 0.1), 2 * x.nbytes),
        19: (lambda: op.op019_sigmoid(x), lambda: torch.sigmoid(x), 2 * x.nbytes),
        20: (lambda: op.op020_tanh(x), lambda: torch.tanh(x), 2 * x.nbytes),
        21: (lambda: op.op021_gelu(x), lambda: F.gelu(x, approximate="tanh"), 2 * x.nbytes),
        22: (lambda: op.op022_silu(x), lambda: F.silu(x), 2 * x.nbytes),
        23: (lambda: op.op023_softplus(x), lambda: F.softplus(x), 2 * x.nbytes),
        24: (lambda: op.op024_elu(x, 0.7), lambda: F.elu(x, 0.7), 2 * x.nbytes),
        25: (lambda: op.op025_hard_sigmoid(x), lambda: F.hardsigmoid(x), 2 * x.nbytes),
        26: (lambda: op.op026_hard_swish(x), lambda: F.hardswish(x), 2 * x.nbytes),
        27: (lambda: op.op027_square(x), lambda: torch.square(x), 2 * x.nbytes),
        28: (lambda: op.op028_sqrt(unary_x), lambda: torch.sqrt(unary_x), 2 * x.nbytes),
        29: (lambda: op.op029_exp(x), lambda: torch.exp(x), 2 * x.nbytes),
        30: (lambda: op.op030_log(unary_x), lambda: torch.log(unary_x), 2 * x.nbytes),
        31: (lambda: op.op031_clamp(x, -1.0, 1.0), lambda: x.clamp(-1.0, 1.0), 2 * x.nbytes),
        32: (lambda: op.op032_where(x > 0, x, y), lambda: torch.where(x > 0, x, y), 4 * x.nbytes),
    }
    if number in mapping:
        return mapping[number]
    rows = max(1, x.numel() // 256)
    a = x[:rows * 256].reshape(rows, 256)
    bias = torch.randn(256, device=x.device)
    return lambda: op.op033_row_bias_add(a, bias), lambda: a + bias, 3 * a.nbytes


def _make_reduction_case(op, number, size, device):
    rows, cols = _matrix_rows(size), _matrix_cols(size)
    a = torch.randn(rows, cols, device=device)
    gamma = torch.randn(cols, device=device)
    beta = torch.randn(cols, device=device)
    mean = torch.randn(cols, device=device)
    var = torch.rand(cols, device=device) + 0.2
    group_x = torch.randn(rows, 4, cols // 4, device=device)
    group_gamma = torch.randn(4, device=device)
    group_beta = torch.randn(4, device=device)
    mapping = {
        34: (lambda: op.op034_row_sum(a), lambda: a.float().sum(-1)),
        35: (lambda: op.op035_row_mean(a), lambda: a.float().mean(-1)),
        36: (lambda: op.op036_row_max(a), lambda: a.max(-1).values),
        37: (lambda: op.op037_row_min(a), lambda: a.min(-1).values),
        38: (lambda: op.op038_l1_norm(a), lambda: a.float().abs().sum(-1)),
        39: (lambda: op.op039_l2_norm(a), lambda: torch.linalg.vector_norm(a.float(), dim=-1)),
        40: (lambda: op.op040_variance(a), lambda: a.float().var(-1, correction=0)),
        41: (lambda: op.op041_argmax(a), lambda: a.argmax(-1)),
        42: (lambda: op.op042_log_softmax(a), lambda: a.float().log_softmax(-1)),
        43: (lambda: op.op043_rms_norm(a, gamma), lambda: a * torch.rsqrt(a.square().mean(-1, keepdim=True) + 1e-5) * gamma),
        44: (lambda: op.op044_batch_norm_inference(a, mean, var, gamma, beta), lambda: (a - mean) * torch.rsqrt(var + 1e-5) * gamma + beta),
        45: (lambda: op.op045_group_norm(group_x, group_gamma, group_beta, 4), lambda: F.group_norm(group_x, 4, group_gamma, group_beta, 1e-5)),
        46: (lambda: op.op046_cumsum(a), lambda: a.cumsum(-1)),
    }
    return _none_case(*mapping[number])


def _make_indexing_sparse_case(op, number, size, device):
    rows = {"small": 16, "medium": 64, "large": 128}[size]
    cols = 64
    x = torch.randn(rows, cols, device=device)
    weight = torch.randn(rows + 8, cols, device=device)
    if number == 47:
        idx = torch.arange(rows, device=device, dtype=torch.int64) % weight.shape[0]
        return _none_case(lambda: op.op047_embedding(weight, idx), lambda: F.embedding(idx, weight))
    if number == 48:
        idx = torch.arange(rows * 8, device=device, dtype=torch.int64).reshape(rows, 8) % cols
        return _none_case(lambda: op.op048_gather_rows(x, idx), lambda: torch.gather(x, 1, idx))
    if number == 49:
        idx = torch.arange(rows * 8, device=device, dtype=torch.int64).reshape(rows, 8) % cols
        src = torch.randn(rows, 8, device=device)
        out = torch.zeros_like(x)
        return _none_case(lambda: op.op049_scatter_rows_(out.clone(), idx, src), lambda: out.clone().scatter_(1, idx, src))
    if number == 50:
        idx = torch.arange(rows * 8, device=device, dtype=torch.int64).reshape(rows, 8) % cols
        src = torch.randn(rows, 8, device=device)
        out = torch.zeros_like(x)
        return _none_case(lambda: op.op050_scatter_add_rows_(out.clone(), idx, src), lambda: out.clone().scatter_add_(1, idx, src))
    if number == 51:
        idx = torch.arange(rows, device=device, dtype=torch.int64) % rows
        return _none_case(lambda: op.op051_index_select_rows(x, idx), lambda: torch.index_select(x, 0, idx))
    if number == 52:
        idx = torch.arange(rows, device=device, dtype=torch.int64) % cols
        return _none_case(lambda: op.op052_one_hot(idx, cols), lambda: F.one_hot(idx, cols).float())
    if number == 53:
        return _none_case(lambda: op.op053_row_topk(x, 8), lambda: torch.topk(x, 8, dim=-1))
    if number == 54:
        nnz_per_row = 4
        row_ptr = torch.arange(rows + 1, device=device, dtype=torch.int32) * nnz_per_row
        col = (torch.arange(rows * nnz_per_row, device=device, dtype=torch.int32) % cols)
        values = torch.randn(rows * nnz_per_row, device=device)
        vector = torch.randn(cols, device=device)
        sparse = torch.sparse_csr_tensor(
            row_ptr.long(), col.long(), values, size=(rows, cols), device=device
        )
        ref = lambda: torch.mv(sparse, vector)
        return _none_case(lambda: op.op054_csr_spmv(row_ptr, col, values, vector, nnz_per_row), ref)
    if number == 55:
        idx = torch.arange(rows * 4, device=device, dtype=torch.int32) % cols
        values = torch.randn(rows * 4, device=device)
        ref = lambda: torch.zeros(cols, device=device).index_add_(0, idx.long(), values)
        return _none_case(lambda: op.op055_coo_scatter_add(idx, values, cols), ref)
    idx = torch.arange(rows * 4, device=device, dtype=torch.int32) % weight.shape[0]
    offsets = torch.arange(rows + 1, device=device, dtype=torch.int32) * 4
    ref = lambda: F.embedding_bag(
        idx.long(), weight, offsets[:-1].long(), mode="sum"
    )
    return _none_case(lambda: op.op056_embedding_bag_sum(weight, idx, offsets, 4), ref)


def _make_linear_algebra_case(op, number, size, device):
    side = _side(size)
    if number == 57:
        a = torch.randn(4, side, side, device=device, dtype=torch.float16)
        b = torch.randn_like(a)
        return _none_case(lambda: op.op057_batched_matmul(a, b), lambda: torch.bmm(a, b).float())
    if number == 58:
        a = torch.randn(side, side, device=device)
        x = torch.randn(side, device=device)
        return _none_case(lambda: op.op058_gemv(a, x), lambda: a @ x)
    if number == 59:
        a = torch.randn(side, device=device)
        b = torch.randn(side, device=device)
        return _none_case(lambda: op.op059_outer_product(a, b), lambda: torch.outer(a, b))
    x = torch.randn(side, side, device=device, dtype=torch.float16)
    weight = torch.randn(side, side, device=device, dtype=torch.float16)
    bias = torch.randn(side, device=device, dtype=torch.float16)
    return _none_case(lambda: op.op060_linear_bias(x, weight, bias), lambda: F.linear(x, weight, bias).float())


def _make_vision_case(op, number, size, device):
    spatial = {"small": 16, "medium": 32, "large": 64}[size]
    x = torch.randn(2, 4, spatial, spatial, device=device)
    if number == 61:
        x1 = torch.randn(2, 4, spatial, device=device)
        w = torch.randn(8, 4, 3, device=device)
        b = torch.randn(8, device=device)
        return _none_case(lambda: op.op061_conv1d(x1, w, b, stride=1, padding=1), lambda: F.conv1d(x1, w, b, padding=1))
    if number == 62:
        w = torch.randn(8, 4, 3, 3, device=device)
        b = torch.randn(8, device=device)
        return _none_case(lambda: op.op062_conv2d(x, w, b, padding=1), lambda: F.conv2d(x, w, b, padding=1))
    if number == 63:
        w = torch.randn(4, 1, 3, 3, device=device)
        b = torch.randn(4, device=device)
        return _none_case(lambda: op.op063_depthwise_conv2d(x, w, b, padding=1), lambda: F.conv2d(x, w, b, padding=1, groups=4))
    if number == 64:
        w = torch.randn(8, 4, device=device)
        b = torch.randn(8, device=device)
        return _none_case(lambda: op.op064_pointwise_conv2d(x, w, b), lambda: F.conv2d(x, w[:, :, None, None], b))
    if number == 65:
        return _none_case(lambda: op.op065_max_pool2d(x, (3, 3), 2, 1), lambda: F.max_pool2d(x, 3, 2, 1))
    if number == 66:
        return _none_case(lambda: op.op066_avg_pool2d(x, (3, 3), 2, 1), lambda: F.avg_pool2d(x, 3, 2, 1, count_include_pad=False))
    if number == 67:
        return _none_case(lambda: op.op067_nearest_resize(x, spatial * 2, spatial * 2), lambda: F.interpolate(x, (spatial * 2, spatial * 2), mode="nearest"))
    return _none_case(lambda: op.op068_bilinear_resize(x, spatial * 2, spatial * 2), lambda: F.interpolate(x, (spatial * 2, spatial * 2), mode="bilinear", align_corners=False))


def _make_transformer_case(op, number, size, device):
    length = {"small": 16, "medium": 64, "large": 128}[size]
    dim = 64
    if number == 69:
        q = torch.randn(2, 2, length, dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        return _none_case(lambda: op.op069_causal_attention(q, k, v), lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    if number == 70:
        x = torch.randn(length, 4, dim, device=device)
        theta = torch.randn(length, dim // 2, device=device)
        cos, sin = theta.cos(), theta.sin()
        def ref():
            out = torch.empty_like(x)
            out[..., 0::2] = x[..., 0::2] * cos[:, None] - x[..., 1::2] * sin[:, None]
            out[..., 1::2] = x[..., 0::2] * sin[:, None] + x[..., 1::2] * cos[:, None]
            return out
        return _none_case(lambda: op.op070_rope(x, cos, sin), ref)
    if number == 71:
        scores = torch.randn(2, 4, length, length, device=device)
        slopes = torch.randn(4, device=device)
        q = torch.arange(length, device=device)[:, None]
        k = torch.arange(length, device=device)[None, :]
        return _none_case(lambda: op.op071_alibi_bias(scores, slopes), lambda: scores + slopes[None, :, None, None] * (k - q))
    if number == 72:
        new_k = torch.randn(2, 2, 4, dim, device=device)
        new_v = torch.randn_like(new_k)
        positions = torch.arange(4, device=device, dtype=torch.int64).repeat(2, 1)
        ck = torch.zeros(2, 2, length, dim, device=device)
        cv = torch.zeros_like(ck)
        def ref():
            ek, ev = ck.clone(), cv.clone()
            for b in range(2):
                for t in range(4):
                    ek[b, :, positions[b, t]] = new_k[b, :, t]
                    ev[b, :, positions[b, t]] = new_v[b, :, t]
            return ek, ev
        return _none_case(lambda: op.op072_kv_cache_append(new_k, new_v, positions, ck.clone(), cv.clone()), ref)
    if number == 73:
        page_size = 16
        pages = 8
        q = torch.randn(2, 2, dim, device=device)
        k_cache = torch.randn(pages, page_size, 2, dim, device=device)
        v_cache = torch.randn_like(k_cache)
        table = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=device, dtype=torch.int32)
        lengths = torch.tensor([min(length, 48), min(length, 64)], device=device, dtype=torch.int32)
        def ref():
            out = torch.empty_like(q, dtype=torch.float32)
            for b in range(2):
                logical_k = torch.cat([k_cache[table[b, p]] for p in range(4)], dim=0)[:lengths[b]]
                logical_v = torch.cat([v_cache[table[b, p]] for p in range(4)], dim=0)[:lengths[b]]
                score = torch.einsum("hd,thd->ht", q[b], logical_k) / math.sqrt(dim)
                out[b] = torch.einsum("ht,thd->hd", score.softmax(-1), logical_v)
            return out
        return _none_case(lambda: op.op073_paged_attention(q, k_cache, v_cache, table, lengths, min(length, 64)), ref)
    if number in (74, 75):
        a = torch.randn(length, dim, device=device)
        residual = torch.randn_like(a)
        gamma = torch.randn(dim, device=device)
        beta = torch.randn(dim, device=device)
        if number == 74:
            return _none_case(lambda: op.op074_residual_layer_norm(a, residual, gamma, beta), lambda: F.layer_norm(a + residual, (dim,), gamma, beta))
        return _none_case(lambda: op.op075_residual_rms_norm(a, residual, gamma), lambda: (a + residual) * torch.rsqrt((a + residual).square().mean(-1, keepdim=True) + 1e-5) * gamma)
    if number in (76, 77):
        gate = torch.randn(_shape(size), device=device)
        value = torch.randn_like(gate)
        if number == 76:
            return lambda: op.op076_swiglu(gate, value), lambda: F.silu(gate) * value, 3 * gate.nbytes
        return lambda: op.op077_geglu(gate, value), lambda: F.gelu(gate, approximate="tanh") * value, 3 * gate.nbytes
    if number == 78:
        ids = torch.arange(_shape(size), device=device, dtype=torch.int64) % 16
        return _none_case(lambda: op.op078_moe_expert_count(ids, 16), lambda: torch.bincount(ids, minlength=16).int())
    if number == 79:
        logits = torch.randn(_matrix_rows(size), _matrix_cols(size), device=device)
        return lambda: op.op079_temperature_scale(logits, 0.7), lambda: logits / 0.7, 2 * logits.nbytes
    if number == 80:
        logits = torch.randn(_matrix_rows(size), _matrix_cols(size), device=device)
        return _none_case(lambda: op.op080_greedy_decode(logits), lambda: logits.argmax(-1))
    logits = torch.randn(_matrix_rows(size), _matrix_cols(size), device=device)
    token_ids = (torch.arange(_matrix_rows(size) * 8, device=device, dtype=torch.int64).reshape(_matrix_rows(size), 8) % _matrix_cols(size))
    def ref():
        out = logits.clone()
        chosen = out.gather(1, token_ids)
        out.scatter_(1, token_ids, torch.where(chosen > 0, chosen / 1.2, chosen * 1.2))
        return out
    return _none_case(lambda: op.op081_repetition_penalty_(logits.clone(), token_ids, 1.2), ref)


def _make_quantization_case(op, number, size, device):
    n = _shape(size)
    if number in (82, 83, 87, 88):
        count = n + 3 if number == 88 else n
        x = torch.randn(count, device=device) * 3
        scale = torch.tensor(0.05, device=device)
    if number == 82:
        return _none_case(lambda: op.op082_per_tensor_quantize(x, scale), lambda: _quantize_reference(x, scale))
    if number == 83:
        q = _quantize_reference(x, scale)
        return lambda: op.op083_per_tensor_dequantize(q, scale), lambda: q.float() * scale, q.nbytes + x.nbytes
    if number == 84:
        a = torch.randn(_matrix_rows(size), _matrix_cols(size), device=device)
        ref = lambda: (_quantize_reference(a, a.abs().amax(-1, keepdim=True).div(127).clamp_min(1e-12)), a.abs().amax(-1).div(127).clamp_min(1e-12))
        return _none_case(lambda: op.op084_per_row_quantize(a), ref)
    if number == 85:
        a = torch.randn(_matrix_rows(size), _matrix_cols(size), device=device)
        ref = lambda: (_quantize_reference(a, a.abs().amax(0, keepdim=True).div(127).clamp_min(1e-12)), a.abs().amax(0).div(127).clamp_min(1e-12))
        return _none_case(lambda: op.op085_per_channel_quantize(a), ref)
    if number == 86:
        side = _side(size)
        a = torch.randint(-8, 8, (side, side), device=device, dtype=torch.int8)
        b = torch.randint(-8, 8, (side, side), device=device, dtype=torch.int8)
        sa = torch.tensor(0.1, device=device)
        sb = torch.tensor(0.2, device=device)
        return _none_case(lambda: op.op086_int8_matmul(a, b, sa, sb), lambda: (a.int() @ b.int()).float() * sa * sb)
    if number == 87:
        return lambda: op.op087_fake_quantize(x, scale), lambda: _quantize_reference(x, scale).float() * scale, 2 * x.nbytes
    # Deliberately include a partial final block.  Padding with zeros is valid
    # for abs-max and lets the reference stay vectorized.
    def ref_blockwise():
        blocks = (x.numel() + 255) // 256
        padded = F.pad(x, (0, blocks * 256 - x.numel()))
        scales = padded.reshape(blocks, 256).abs().amax(-1).div(127).clamp_min(1e-12)
        expanded = torch.repeat_interleave(scales, 256)[:x.numel()]
        return _quantize_reference(x, expanded), scales
    return _none_case(lambda: op.op088_blockwise_quantize(x, 256), ref_blockwise)


def _make_loss_optimizer_case(op, number, size, device):
    rows, cols = _matrix_rows(size), _matrix_cols(size)
    if number <= 93:
        a = torch.randn(rows, cols, device=device)
        b = torch.randn_like(a)
    if number == 89:
        return _none_case(lambda: op.op089_mse(a, b), lambda: (a - b).square().mean(-1))
    if number == 90:
        target = torch.rand_like(a)
        return _none_case(lambda: op.op090_bce_with_logits(a, target), lambda: F.binary_cross_entropy_with_logits(a, target, reduction="none"))
    if number == 91:
        labels = torch.arange(rows, device=device, dtype=torch.int64) % cols
        return _none_case(lambda: op.op091_cross_entropy(a, labels), lambda: F.cross_entropy(a, labels, reduction="none"))
    if number == 92:
        return _none_case(lambda: op.op092_cosine_similarity(a, b), lambda: F.cosine_similarity(a, b, dim=-1))
    if number == 93:
        log_p, log_q = a.log_softmax(-1), b.log_softmax(-1)
        return _none_case(lambda: op.op093_kl_divergence(log_p, log_q), lambda: (log_p.exp() * (log_p - log_q)).sum(-1))

    p = torch.randn(_shape(size), device=device)
    g = torch.randn_like(p)
    if number == 94:
        def tri_sgd():
            param = p.clone()
            return op.op094_sgd_(param, g, 0.03)
        def ref_sgd():
            param = p.clone()
            param.add_(g, alpha=-0.03)
            return param
        return tri_sgd, ref_sgd, 2 * p.nbytes
    if number == 95:
        velocity = torch.randn_like(p)
        def tri_momentum():
            param, state = p.clone(), velocity.clone()
            op.op095_momentum_sgd_(param, g, state, 0.03, 0.9)
            return param, state
        def ref_momentum():
            param, state = p.clone(), velocity.clone()
            state.mul_(0.9).add_(g)
            param.add_(state, alpha=-0.03)
            return param, state
        return _none_case(tri_momentum, ref_momentum)
    if number == 96:
        m = torch.randn_like(p)
        v = torch.rand_like(p)
        def tri_adam():
            param, first, second = p.clone(), m.clone(), v.clone()
            op.op096_adam_(param, g, first, second, 3)
            return param, first, second
        def ref_adam():
            param, first, second = p.clone(), m.clone(), v.clone()
            first.mul_(0.9).add_(g, alpha=0.1)
            second.mul_(0.999).addcmul_(g, g, value=0.001)
            update = (first / (1 - 0.9 ** 3)) / (torch.sqrt(second / (1 - 0.999 ** 3)) + 1e-8)
            param.add_(update, alpha=-1e-3)
            return param, first, second
        return _none_case(tri_adam, ref_adam)
    if number == 97:
        m = torch.randn_like(p)
        v = torch.rand_like(p)
        def tri_adamw():
            param, first, second = p.clone(), m.clone(), v.clone()
            op.op097_adamw_(param, g, first, second, 3)
            return param, first, second
        def ref_adamw():
            param, first, second = p.clone(), m.clone(), v.clone()
            old_param = param.clone()
            first.mul_(0.9).add_(g, alpha=0.1)
            second.mul_(0.999).addcmul_(g, g, value=0.001)
            update = (first / (1 - 0.9 ** 3)) / (torch.sqrt(second / (1 - 0.999 ** 3)) + 1e-8)
            update.add_(old_param, alpha=0.01)
            param.add_(update, alpha=-1e-3)
            return param, first, second
        return _none_case(tri_adamw, ref_adamw)
    if number == 98:
        state = torch.rand_like(p)
        def tri_adagrad():
            param, accumulator = p.clone(), state.clone()
            op.op098_adagrad_(param, g, accumulator, 0.02)
            return param, accumulator
        def ref_adagrad():
            param, accumulator = p.clone(), state.clone()
            accumulator.addcmul_(g, g)
            param.addcdiv_(g, accumulator.sqrt().add_(1e-10), value=-0.02)
            return param, accumulator
        return _none_case(tri_adagrad, ref_adagrad)
    if number == 99:
        norm = g.norm().item()
        def tri_clip():
            grad = g.clone()
            return op.op099_gradient_norm_clip_(grad, norm, 0.5)
        def ref_clip():
            grad = g.clone()
            grad.mul_(min(1.0, 0.5 / (norm + 1e-6)))
            return grad
        return tri_clip, ref_clip, 2 * g.nbytes
    return _none_case(lambda: op.op100_non_finite_check(p), lambda: torch.isfinite(p).logical_not().any().int())
