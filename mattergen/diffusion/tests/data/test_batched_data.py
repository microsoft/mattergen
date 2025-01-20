# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from mattergen.diffusion.data.batched_data import collate_fn


def test_collate_fn():
    """Collate two pieces of data"""
    data1 = {"a": torch.tensor([1, 2, 3, 4]), "b": torch.tensor([[1, 2, 3]]), "name": "data1"}
    data2 = {"a": torch.tensor([10, 11]), "b": torch.tensor([[10, 11, 12]]), "name": "data2"}
    collated = collate_fn([data1, data2], dense_field_names=["b"])
    assert collated["a"].tolist() == [1, 2, 3, 4, 10, 11]
    assert collated["b"].tolist() == [[1, 2, 3], [10, 11, 12]]
    assert collated["name"] == ["data1", "data2"]
    assert collated.get_batch_idx("a").tolist() == [0, 0, 0, 0, 1, 1]
    assert collated.get_batch_idx("b") is None
    assert collated.get_batch_idx("name") is None


def test_to_data_list():
    """Collate and then unpack two pieces of data."""
    data1 = {"a": torch.tensor([1, 2, 3, 4]), "b": torch.tensor([[1, 2, 3]]), "name": "data1"}
    data2 = {"a": torch.tensor([10, 11]), "b": torch.tensor([[10, 11, 12]]), "name": "data2"}
    collated = collate_fn([data1, data2], dense_field_names=["b"])
    data_list = collated.to_data_list()
    assert data_list[0]["a"].tolist() == [1, 2, 3, 4]
    assert data_list[0]["b"].tolist() == [[1, 2, 3]]
    assert data_list[0]["name"] == "data1"
    assert data_list[1]["a"].tolist() == [10, 11]
    assert data_list[1]["b"].tolist() == [[10, 11, 12]]
    assert data_list[1]["name"] == "data2"
