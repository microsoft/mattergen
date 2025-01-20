# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Iterable, Protocol

import torch
from torch_scatter import scatter

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.score_models.base import Diffusable


class Metric(Protocol):
    """
    Computes a metric to be logged during training.
    Each metric must have a name which is used as a prefix for the metric in the log.
    """

    name: str

    def __call__(
        self,
        *,
        loss_per_sample_per_field: Dict[str, torch.Tensor],
        multi_corruption: MultiCorruption,
        score_model_output: Diffusable,
        t: torch.Tensor,
        batch_idx: Dict[str, torch.LongTensor],
        batch: BatchedData,
        noisy_batch: BatchedData,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes a metric to be logged during training. Useful, e.g., for plotting loss over time.

        Args:
            loss_per_sample_per_field: Dict[str, torch.Tensor], where each tensor has shape (batch_size,).
            multi_corruption: MultiCorruption
            score_model_output: the output produced by the model per field.
            t: shape (batch_size,). Time for each element in the loss.
            batch_idx: Dict[str, torch.LongTensor]: batch indices per field
            batch: BatchedData: the clean (un-perturbed) batched data
            noisy_batch: BatchedData: the corrupted batched data
        """
        pass


def loss_per_time_bin(
    loss_per_sample: torch.Tensor, t: torch.Tensor, bins: torch.Tensor
) -> torch.Tensor:
    """
    Aggregate loss per bin. Useful for plotting loss over time.

    Args:
        loss_per_sample: shape (batch_size,). Loss for each sample.
        t: shape (batch_size,). Time for each element in the loss.
        bins: shape (num_bins,). Upper boundaries of the time bins.
    Returns:
        avg_loss_per_bin: shape (num_bins,). Average loss per time bin.
    """
    bin_per_element = torch.bucketize(t, bins)
    avg_loss_per_bin = scatter(
        src=loss_per_sample, index=bin_per_element, dim_size=bins.shape[0], reduce="mean"
    )
    return avg_loss_per_bin


class LossPerTimeBin(Metric):
    name = "loss_per_time_bin"

    def __init__(self, t_min: float = 0.0, t_max: float = 1.0, num_bins: int = 10):
        self.bins = torch.linspace(t_min, t_max, num_bins + 1)

    def __call__(
        self,
        *,
        loss_per_sample_per_field: Dict[str, torch.Tensor],
        t: torch.Tensor,
        **_,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss bins per diffusion time bin. Useful for plotting loss over diffusion time.
        """
        metrics_dict = {}
        for k, v in loss_per_sample_per_field.items():
            assert v.shape == t.shape

            # first bin is always empty because no time is less than t_min, so we skip it
            avg_loss_per_bin = loss_per_time_bin(
                loss_per_sample_per_field[k],
                t,
                bins=self.bins.to(loss_per_sample_per_field[k].device)[1:],
            )
            metrics_dict.update(
                {
                    f"{k}_{self.bins[ix]:.2f}-{self.bins[ix + 1]:.2f}": avg_loss_per_bin[ix]
                    for ix in range(len(avg_loss_per_bin))
                    if avg_loss_per_bin[ix] > 0.0
                }
            )
        return metrics_dict


class MetricsCalculator:
    """
    Computes a set of metrics to be logged during training.
    """

    def __init__(self, metric_fns: Iterable[Metric]):
        self.metric_fns = metric_fns

    def __call__(
        self,
        *,
        loss_per_sample_per_field: Dict[str, torch.Tensor],
        multi_corruption: MultiCorruption,
        score_model_output: torch.Tensor,
        t: torch.Tensor,
        batch_idx: Dict[str, torch.LongTensor],
        batch: BatchedData,
        noisy_batch: BatchedData,
    ) -> Dict[str, torch.Tensor]:
        metrics_dict = {}
        for metric_fn in self.metric_fns:
            _metrics_dict = metric_fn(
                loss_per_sample_per_field=loss_per_sample_per_field,
                multi_corruption=multi_corruption,
                score_model_output=score_model_output,
                t=t,
                batch_idx=batch_idx,
                batch=batch,
                noisy_batch=noisy_batch,
            )
            # prepend metric name to each metric
            metrics_dict.update({f"{metric_fn.name}_{k}": v for k, v in _metrics_dict.items()})
        return metrics_dict
