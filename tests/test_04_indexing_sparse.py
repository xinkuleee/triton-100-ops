"""Reference tests for indexing and sparse operators 047--056."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from _utils import assert_close, load_category, require_gpu

require_gpu(pytest, torch)
ops = load_category("04_indexing_sparse")
DEVICE = "cuda"


def test_op047_embedding():
    weight = torch.randn(19, 17, device=DEVICE)
    indices = torch.tensor([3, 1, 18, 3], device=DEVICE, dtype=torch.int64)
    assert_close(torch, ops.op047_embedding(weight, indices), F.embedding(indices, weight))


def test_op048_gather_rows():
    x = torch.randn(5, 13, device=DEVICE)
    indices = torch.tensor([[0, 3, 12], [1, 4, 7], [2, 5, 8], [3, 6, 9], [4, 7, 10]], device=DEVICE)
    assert_close(torch, ops.op048_gather_rows(x, indices), torch.gather(x, 1, indices))


def test_op049_and_op050_scatter():
    indices = torch.tensor([[0, 2, 4], [1, 3, 5]], device=DEVICE)
    src = torch.randn(2, 3, device=DEVICE)
    out = torch.zeros(2, 7, device=DEVICE)
    expected = out.clone().scatter_(1, indices, src)
    assert_close(torch, ops.op049_scatter_rows_(out.clone(), indices, src), expected)
    duplicate = torch.tensor([[1, 1, 2], [0, 0, 0]], device=DEVICE)
    expected_add = out.clone().scatter_add_(1, duplicate, src)
    assert_close(torch, ops.op050_scatter_add_rows_(out.clone(), duplicate, src), expected_add)


def test_op051_index_select_and_op052_one_hot():
    x = torch.randn(8, 11, device=DEVICE)
    index = torch.tensor([7, 2, 2, 0], device=DEVICE)
    assert_close(torch, ops.op051_index_select_rows(x, index), torch.index_select(x, 0, index))
    assert_close(torch, ops.op052_one_hot(index, 9), F.one_hot(index, 9).float())


def test_op053_row_topk():
    x = torch.randn(7, 31, device=DEVICE)
    values, indices = ops.op053_row_topk(x, 5)
    expected_values, expected_indices = torch.topk(x, 5, dim=-1)
    assert_close(torch, values, expected_values)
    assert torch.equal(indices, expected_indices)


def test_op054_csr_spmv():
    row_ptr = torch.tensor([0, 2, 2, 5], device=DEVICE, dtype=torch.int32)
    col = torch.tensor([0, 3, 1, 2, 3], device=DEVICE, dtype=torch.int32)
    values = torch.randn(5, device=DEVICE)
    vector = torch.randn(4, device=DEVICE)
    expected = torch.stack([
        values[0] * vector[0] + values[1] * vector[3],
        torch.tensor(0.0, device=DEVICE),
        values[2] * vector[1] + values[3] * vector[2] + values[4] * vector[3],
    ])
    assert_close(torch, ops.op054_csr_spmv(row_ptr, col, values, vector, 3), expected)


def test_op055_coo_scatter_add():
    index = torch.tensor([0, 2, 2, 4], device=DEVICE, dtype=torch.int32)
    values = torch.randn(4, device=DEVICE)
    expected = torch.zeros(6, device=DEVICE).index_add_(0, index.long(), values)
    assert_close(torch, ops.op055_coo_scatter_add(index, values, 6), expected)


def test_op056_embedding_bag_sum():
    weight = torch.randn(23, 16, device=DEVICE)
    indices = torch.tensor([1, 3, 5, 2, 2, 7], device=DEVICE, dtype=torch.int32)
    offsets = torch.tensor([0, 3, 4, 6], device=DEVICE, dtype=torch.int32)
    expected = torch.stack([weight[indices[0:3].long()].sum(0), weight[2], weight[indices[4:6].long()].sum(0)])
    assert_close(torch, ops.op056_embedding_bag_sum(weight, indices, offsets, 3), expected)


def test_invalid_dense_indices_follow_masked_contract():
    weight = torch.arange(15, device=DEVICE, dtype=torch.float32).reshape(5, 3)
    indices = torch.tensor([-1, 0, 5], device=DEVICE)
    expected_rows = torch.stack((torch.zeros(3, device=DEVICE), weight[0], torch.zeros(3, device=DEVICE)))
    assert_close(torch, ops.op047_embedding(weight, indices), expected_rows)
    assert_close(torch, ops.op051_index_select_rows(weight, indices), expected_rows)

    x = torch.arange(12, device=DEVICE, dtype=torch.float32).reshape(3, 4)
    gather_indices = torch.tensor([[-1, 0], [3, 4], [1, 2]], device=DEVICE)
    expected_gather = torch.tensor([[0, 0], [7, 0], [9, 10]], device=DEVICE, dtype=torch.float32)
    assert_close(torch, ops.op048_gather_rows(x, gather_indices), expected_gather)
    assert_close(torch, ops.op052_one_hot(indices, 5), (indices[:, None] == torch.arange(5, device=DEVICE)).float())


def test_scatter_ignores_invalid_indices_and_empty_updates():
    out = torch.arange(8, device=DEVICE, dtype=torch.float32).reshape(2, 4)
    indices = torch.tensor([[-1, 2, 4], [0, 3, 9]], device=DEVICE)
    src = torch.full((2, 3), 10.0, device=DEVICE)
    expected = out.clone()
    expected[0, 2] = 10.0
    expected[1, 0] = 10.0
    expected[1, 3] = 10.0
    assert_close(torch, ops.op049_scatter_rows_(out.clone(), indices, src), expected)

    expected_add = out.clone()
    expected_add[0, 2] += 10.0
    expected_add[1, 0] += 10.0
    expected_add[1, 3] += 10.0
    assert_close(torch, ops.op050_scatter_add_rows_(out.clone(), indices, src), expected_add)
    empty = torch.empty((2, 0), device=DEVICE, dtype=torch.int64)
    empty_src = torch.empty((2, 0), device=DEVICE)
    assert_close(torch, ops.op049_scatter_rows_(out.clone(), empty, empty_src), out)


def test_topk_ties_sparse_structure_and_empty_boundaries():
    x = torch.tensor([[4.0, 4.0, 3.0, 4.0]], device=DEVICE)
    values, indices = ops.op053_row_topk(x, 3)
    assert_close(torch, values, torch.tensor([[4.0, 4.0, 4.0]], device=DEVICE))
    assert torch.equal(indices, torch.tensor([[0, 1, 3]], device=DEVICE))

    with pytest.raises(ValueError):
        ops.op054_csr_spmv(
            torch.tensor([0, 2, 1], device=DEVICE, dtype=torch.int32),
            torch.tensor([0], device=DEVICE, dtype=torch.int32),
            torch.ones(1, device=DEVICE),
            torch.ones(1, device=DEVICE),
            2,
        )
    empty_index = torch.empty(0, device=DEVICE, dtype=torch.int32)
    empty_values = torch.empty(0, device=DEVICE)
    assert ops.op055_coo_scatter_add(empty_index, empty_values, 0).shape == (0,)

    offsets = torch.tensor([0, 0, 0], device=DEVICE, dtype=torch.int32)
    bags = ops.op056_embedding_bag_sum(
        torch.ones(3, 5, device=DEVICE), empty_index, offsets, 1
    )
    assert_close(torch, bags, torch.zeros(2, 5, device=DEVICE))


def test_sparse_invalid_columns_are_ignored():
    row_ptr = torch.tensor([0, 3], device=DEVICE, dtype=torch.int32)
    columns = torch.tensor([-1, 1, 7], device=DEVICE, dtype=torch.int32)
    values = torch.tensor([100.0, 2.0, 100.0], device=DEVICE)
    vector = torch.tensor([3.0, 4.0], device=DEVICE)
    assert_close(
        torch,
        ops.op054_csr_spmv(row_ptr, columns, values, vector, 3),
        torch.tensor([8.0], device=DEVICE),
    )

    coo_indices = torch.tensor([-1, 1, 1, 4], device=DEVICE, dtype=torch.int32)
    coo_values = torch.tensor([50.0, 2.0, 3.0, 50.0], device=DEVICE)
    assert_close(
        torch,
        ops.op055_coo_scatter_add(coo_indices, coo_values, 3),
        torch.tensor([0.0, 5.0, 0.0], device=DEVICE),
    )

    weight = torch.arange(12, device=DEVICE, dtype=torch.float32).reshape(3, 4)
    bag_indices = torch.tensor([-1, 1, 3], device=DEVICE, dtype=torch.int32)
    offsets = torch.tensor([0, 3], device=DEVICE, dtype=torch.int32)
    assert_close(
        torch, ops.op056_embedding_bag_sum(weight, bag_indices, offsets, 3), weight[1:2]
    )
