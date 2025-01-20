# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Dict, Literal, Optional, Protocol, Tuple, TypeVar

import torch

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.model_target import ModelTargets
from mattergen.diffusion.training.field_loss import FieldLoss, denoising_score_matching

T = TypeVar("T", bound=BatchedData)


class Loss(Protocol[T]):
    """Loss function for training a score model on multi-field data."""

    def __call__(
        self,
        *,
        multi_corruption: MultiCorruption[T],
        batch: T,
        noisy_batch: T,
        score_model_output: T,
        t: torch.Tensor,
        node_is_unmasked: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pass

    """model_targets tells us what this loss function trains the score model to predict.
    We need this information in order to convert the model output to a score during sampling.
    """
    model_targets: ModelTargets


class SummedFieldLoss(Loss[T]):
    """(Weighted) sum of different loss functions applied on each field."""

    def __init__(
        self,
        loss_fns: Dict[str, FieldLoss],
        model_targets: ModelTargets,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.model_targets = model_targets
        self.loss_fns = loss_fns
        # weights are optional, if not provided, all fields are weighted equally with weight 1.
        if weights is None:
            self.loss_weights = {k: 1.0 for k in self.loss_fns.keys()}
        else:
            assert set(weights.keys()) == set(
                self.loss_fns.keys()
            ), f"weight keys {set(weights.keys())} do not match loss_fns keys {set(self.loss_fns.keys())}"
            self.loss_weights = weights

    def __call__(
        self,
        *,
        multi_corruption: MultiCorruption[T],
        batch: T,
        noisy_batch: T,
        score_model_output: T,
        t: torch.Tensor,
        node_is_unmasked: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_idx = {k: batch.get_batch_idx(k) for k in self.loss_fns.keys()}
        node_is_unmasked = {k: node_is_unmasked for k in self.loss_fns.keys()}

        # Dict[str, torch.Tensor]
        # Keys are field names and values are loss per sample, with shape (batch_size,).
        loss_per_sample_per_field = apply(
            fns=self.loss_fns,
            corruption=multi_corruption.corruptions,
            x=batch,
            noisy_x=noisy_batch,
            score_model_output=score_model_output,
            batch_idx=batch_idx,
            broadcast=dict(t=t, batch_size=batch.get_batch_size(), batch=batch),
            node_is_unmasked=node_is_unmasked,
        )
        assert set([v.shape for v in loss_per_sample_per_field.values()]) == {
            (batch.get_batch_size(),)
        }, "All losses should have shape (batch_size,)."
        # Aggregate losses per field over samples.
        scalar_loss_per_field = {k: v.mean() for k, v in loss_per_sample_per_field.items()}

        # Dict[str, torch.Tensor], dictionary containing metrics to be logged,
        metrics_dict = scalar_loss_per_field
        # This is the loss that is used for backpropagation (after mean aggregation over samples).
        # Shape: (batch_size,)
        agg_loss = torch.stack(
            [self.loss_weights[k] * v for k, v in loss_per_sample_per_field.items()], dim=0
        ).sum(0)

        return (
            agg_loss.mean(),
            metrics_dict,
        )


class DenoisingScoreMatchingLoss(SummedFieldLoss):
    def __init__(
        self,
        model_targets: ModelTargets,
        reduce: Literal["sum", "mean"] = "mean",
        weights: Optional[Dict[str, float]] = None,
        field_center_zero: Optional[Dict[str, bool]] = None,  # Whether to zero center each field.
    ):
        if field_center_zero is not None:
            assert set(field_center_zero.keys()) == set(model_targets.keys())

        super().__init__(
            loss_fns={
                k: partial(
                    denoising_score_matching,
                    reduce=reduce,
                    model_target=v,
                )
                for k, v in model_targets.items()
            },
            model_targets=model_targets,
            weights=weights,
        )
