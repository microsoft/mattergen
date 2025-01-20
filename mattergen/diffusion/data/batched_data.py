# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol, Sequence, TypeVar, runtime_checkable

import torch
from torch_scatter import scatter

T = TypeVar("T")

logger = logging.getLogger(__name__)


@runtime_checkable
class BatchedData(Protocol):
    def replace(self: T, **vals: torch.Tensor) -> T:
        """Return a copy of self with some fields replaced with new values."""

    def get_batch_idx(self, field_name: str) -> torch.LongTensor | None:
        """Get the batch index (i.e., which row belongs to which sample) for a given field.
        For 'dense' type data, where every sample has the same shape and the first dimension is the
        batch dimension, this method should return None. Mathematically,
        returning None will be treated the same as returning a tensor [0, 1, 2, ..., batch_size - 1]
        but I expect memory access in other functions to be more efficient if you return None.
        """

    def get_batch_size(self) -> int:
        """Get the batch size."""

    def device(self) -> torch.device:
        """Get the device of the batch."""

    def __getitem__(self, field_name: str) -> torch.Tensor:
        """Get a field from the batch."""

    def to(self: T, device: torch.device) -> T:
        """Move the batch to a given device."""

    def clone(self: T) -> T:
        """Return a copy with all the tensors cloned."""


@dataclass
class SimpleBatchedData(BatchedData):
    """Implements BatchedData as a pair of mappings from field names to tensors."""

    data: Mapping[str, Any]
    batch_idx: Mapping[str, torch.LongTensor]

    def replace(self, **vals: torch.Tensor) -> SimpleBatchedData:
        """Return a copy of self with some fields of self.data replaced with new values."""
        return replace(self, data=self._updated_data(**vals))

    def get_batch_idx(self, field_name: str) -> torch.LongTensor | None:
        return self.batch_idx[field_name]

    def _updated_data(self, **vals):
        return dict(self.data, **vals)

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    def get_batch_size(self) -> int:
        L = []
        for k, v in self.batch_idx.items():
            if v is None:
                d = self.data[k]
                L.append(d.shape[0] if isinstance(d, torch.Tensor) else len(d))
            else:
                if len(v) == 0:
                    logger.warning(f"Empty batch index for field {k}")
                    L.append(0)
                else:
                    L.append(int(torch.max(v).item()) + 1)
        return max(L)

    @property
    def device(self) -> torch.device:
        # Beware, there are no checks that all values are on the same device
        return next(v.device for v in self.data.values())

    def to(self, device) -> "SimpleBatchedData":
        """Modify self in-place to move all tensors to the given device, and return self"""
        if isinstance(self.data, dict):
            for k in self.data.keys():
                if isinstance(self.data[k], torch.Tensor):
                    self.data[k] = self.data[k].to(device)
        if isinstance(self.batch_idx, dict):
            for key in self.batch_idx.keys():
                if self.batch_idx[key] is None:
                    continue
                self.batch_idx[key] = self.batch_idx[key].to(device)

        return self

    def clone(self) -> SimpleBatchedData:
        return SimpleBatchedData(
            data={
                k: v.clone() if isinstance(v, torch.Tensor) else deepcopy(v)
                for k, v in self.data.items()
            },
            batch_idx={k: v.clone() if v is not None else None for k, v in self.batch_idx.items()},
        )

    def to_data_list(self) -> list[dict[str, torch.Tensor]]:
        """Converts this instance to a list of dictionaries, each of which corresponds to a single datapoint in
        `batched_data`. The keys of the dictionaries match the keys of `batched_data`.
        """

        batch_size = self.get_batch_size()
        if batch_size == 0:
            return []

        def _unpack(k, i):
            if self.batch_idx[k] is not None:
                return self.data[k][self.batch_idx[k] == i]
            elif isinstance(self.data[k], torch.Tensor):
                return self.data[k][i : i + 1]
            else:
                return self.data[k][i]

        return [{k: _unpack(k, i) for k in self.data.keys()} for i in range(batch_size)]


def collate_fn(
    states: list[dict[str, Any]], dense_field_names: Sequence[str] = ()
) -> SimpleBatchedData:
    """
    Combine a list of samples into a SimpleBatchedData object.

    The association between the index in `states[i][k]` and a row in the `batched_data[k]` is
    stored in `batched_data.batch_idx[k]`. If the `k` appears in
    `dense_field_names`, `batched_data.batch_idx[k]` is `None` and the data is
    simply stacked along the first dimension.

    Non-tensor values are put into lists.
    """
    assert states, "Cannot collate empty list"
    concatenated_data = {}
    batch_idx: dict[str, torch.Tensor | None] = {}
    for k, v in states[0].items():
        if isinstance(v, torch.Tensor):
            concatenated_data[k] = torch.cat([x[k] for x in states], dim=0)
            if k in dense_field_names:
                if any(x[k].shape[0] != 1 for x in states):
                    raise ValueError(
                        f"First dimension should be batch dimension. Instead key {k} has shape {states[0][k].shape}"
                    )
                batch_idx[k] = None
            else:
                batch_idx[k] = _construct_batch_idx(states, k)
        else:
            concatenated_data[k] = [x[k] for x in states]
            batch_idx[k] = None

    batch = SimpleBatchedData(data=concatenated_data, batch_idx=batch_idx)
    if "edge_index" in batch.data:
        batch = batch.replace(
            edge_index=_batch_edge_index(
                batch["edge_index"],
                batch.batch_idx["atomic_numbers"],
                batch.batch_idx["edge_index"],
            ),
        )
    return batch


def _batch_edge_index(edge_index, atom_batch_idx, edge_batch_idx):
    num_atoms = scatter(torch.ones_like(atom_batch_idx), atom_batch_idx)
    num_atoms_acc = torch.nn.functional.pad(torch.cumsum(num_atoms, 0)[:-1], [1, 0], "constant", 0)
    return edge_index + num_atoms_acc[edge_batch_idx].unsqueeze(1)


def _construct_batch_idx(data_list: list[Any], field_name: str) -> torch.LongTensor:
    """Construct batch index tensor for one field."""
    batch_size = len(data_list)
    return torch.repeat_interleave(
        torch.arange(0, batch_size),
        torch.tensor([x[field_name].shape[0] for x in data_list]),
    )
