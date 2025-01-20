# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from mattergen.diffusion.data.batched_data import SimpleBatchedData, _batch_edge_index, collate_fn


def test_collate_fn():
    state1 = dict(foo=torch.ones(2, 3), bar=torch.ones(5, 2))
    state2 = dict(foo=torch.zeros(3, 3), bar=torch.zeros(2, 2))
    batch = collate_fn([state1, state2])

    field_names = list(state1.keys())

    expected = SimpleBatchedData(
        data=dict(
            foo=torch.Tensor(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            bar=torch.Tensor(
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
            ),
        ),
        batch_idx={
            "foo": torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
            "bar": torch.tensor([0, 0, 0, 0, 0, 1, 1], dtype=torch.long),
        },
    )

    for k in field_names:
        assert torch.equal(batch[k], expected[k])
        assert torch.equal(batch.get_batch_idx(k), expected.get_batch_idx(k))

    assert batch.get_batch_size() == 2


def test_batch_edge_index():
    edge_index = torch.tensor(
        [[0, 1], [0, 2], [1, 2], [0, 1], [0, 3], [1, 2], [2, 3], [0, 1], [1, 3]]
    )
    atom_batch_idx = torch.tensor([0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    edge_batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2])
    torch.testing.assert_close(
        _batch_edge_index(edge_index, atom_batch_idx, edge_batch_idx),
        torch.tensor([[0, 1], [0, 2], [1, 2], [2, 3], [2, 5], [3, 4], [4, 5], [7, 8], [8, 10]]),
    )
