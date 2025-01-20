# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
)

import torch

from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.corruption.sde_lib import SDE, Corruption
from mattergen.diffusion.data.batched_data import BatchedData

R = TypeVar("R")
Diffusable = TypeVar("Diffusable", bound=BatchedData)


def _first(s: Iterable):
    return next(iter(s))


@dataclass
class MultiCorruptionConfig:
    discrete_corruptions: dict[str, Any] = field(default_factory=dict)
    sdes: dict[str, Any] = field(default_factory=dict)


class MultiCorruption(Generic[Diffusable]):
    """Wraps multiple `Corruption` instances to operate on different fields of a State

    In the forward process, each field of State is corrupted independently.

    In the reverse process, a single score model takes in the entire State and
    uses it to estimate the score with respect to each field of the State.
    """

    def _get_batch_indices(self, batch: Diffusable) -> Dict[str, torch.Tensor]:
        return {k: batch.get_batch_idx(k) for k in self.corrupted_fields}

    def __init__(
        self,
        sdes: Optional[Mapping[str, SDE]] = None,
        discrete_corruptions: Optional[Mapping[str, D3PMCorruption]] = None,
    ):
        """
        Args:
            sdes: mapping from fields of batch to SDE corruption processes
            discrete_corruptions: mapping from fields of batch to discrete corruption processes
        """
        if sdes is None:
            sdes = {}
        if discrete_corruptions is None:
            discrete_corruptions = {}
        assert (
            len(sdes) + len(discrete_corruptions) > 0
        ), "Must have at least one corruption process."

        self._sdes = sdes
        self._discrete_corruptions = discrete_corruptions
        assert (
            set(self._sdes.keys()).intersection(set(self._discrete_corruptions.keys())) == set()
        ), "SDEs and corruptions have overlapping keys."
        self._corruptions: Dict[str, Corruption] = {**self._sdes, **self._discrete_corruptions}
        # Make the dict sorted by key (to prevent mismatching checkpoints):
        self._corruptions = {k: self._corruptions[k] for k in sorted(self._corruptions.keys())}

        # All SDEs must have the same T
        T_vals = [corruption.T for corruption in self.corruptions.values()]
        assert len(set(T_vals)) == 1

    @property
    def sdes(self) -> Mapping[str, SDE]:
        return self._sdes

    @property
    def has_discrete_corruptions(self) -> bool:
        return len(self.discrete_corruptions) > 0

    @property
    def discrete_corruptions(self) -> Mapping[str, Corruption]:
        return self._discrete_corruptions

    @property
    def corruptions(self) -> Mapping[str, Corruption]:
        return self._corruptions

    @property
    def corrupted_fields(self) -> List[str]:
        return list(self.corruptions.keys())

    @cached_property
    def T(self) -> float:
        return _first(self.corruptions.values()).T

    def sample_marginal(self, batch: Diffusable, t) -> Diffusable:
        def fn_getter(corruption: Corruption) -> Callable[..., Tuple[torch.Tensor, torch.Tensor]]:
            return corruption.sample_marginal

        noisy_data = self._apply_corruption_fn(
            fn_getter,
            x=batch,
            batch_idx=self._get_batch_indices(batch),
            broadcast=dict(t=t),
        )
        noisy_batch = batch.replace(**noisy_data)
        return noisy_batch

    def sde(
        self, batch: Diffusable, t: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Get drift and diffusion for each component of the state"""
        assert (
            not self.has_discrete_corruptions
        ), "Cannot call `sde` on a MultiCorruption with non-SDE corruptions"

        fns = {k: sde.sde for k, sde in self.sdes.items()}
        return apply(
            fns=fns,
            broadcast={"batch": batch, "t": t},
            x=batch,
            batch_idx=self._get_batch_indices(batch),
        )

    def _apply_corruption_fn(
        self,
        fn_getter: Callable[[Corruption], Callable[..., R]],
        x: BatchedData,
        batch_idx: Mapping[str, torch.LongTensor],
        broadcast: Optional[Dict] = None,
        apply_to: Optional[Mapping[str, Corruption]] = None,
        **kwargs,
    ) -> Dict[str, R]:
        if apply_to is None:
            apply_to = self.corruptions
        fns = {field_name: fn_getter(corruption) for field_name, corruption in apply_to.items()}
        return apply(
            fns=fns,
            broadcast={**(broadcast or dict()), "batch": x},
            x=x,
            batch_idx=batch_idx,
            **kwargs,
        )


def apply(fns: Dict[str, Callable[..., R]], broadcast, **kwargs) -> Dict[str, R]:
    """Apply different function with different argument values to each field.
    fns: dict of the form {field_name: function_to_apply}
    broadcast: arguments that are identical for every field_name
    kwargs: dict of the form {argument_name: {field_name: argument_value}}
    """
    return {
        field_name: fn(
            **{k: v[field_name] for k, v in kwargs.items() if field_name in v},
            **(broadcast or dict()),
        )
        for field_name, fn in fns.items()
    }
