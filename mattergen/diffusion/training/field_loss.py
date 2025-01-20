# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Literal, Protocol

import torch
from torch_scatter import scatter

from mattergen.diffusion.corruption.corruption import Corruption
from mattergen.diffusion.corruption.sde_lib import maybe_expand
from mattergen.diffusion.d3pm.d3pm import compute_kl_reverse_process
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.discrete_time import to_discrete_time
from mattergen.diffusion.model_target import ModelTarget


def compute_noise_given_sample_and_corruption(
    x: torch.Tensor,
    x_noisy: torch.Tensor,
    corruption: Corruption,
    t: torch.Tensor,
    batch_idx: torch.LongTensor | None,
    batch: BatchedData,
) -> torch.Tensor:
    """
    Recover the (unit-Gaussian-distributed) raw noise that was used to corrupt a batch of samples.
    We first obtain the mean and std of the noisy samples from the corruption via `t` and the clean batch.
    Then we solve:
    x_noisy = x_mean + noise * std w.r.t. `noise`:
    noise = (x_noisy - x_mean) / std
    """
    x_mean, std = corruption.marginal_prob(
        x,
        t=t,
        batch_idx=batch_idx,
        batch=batch,
    )
    return (x_noisy - x_mean) / std


class FieldLoss(Protocol):
    """Loss function for a single field. Because loss functions are defined different ways in different papers,
    we pass loads of keyword arguments. Each loss function will only use a subset of these arguments.
    """

    def __call__(
        self,
        *,
        corruption: Corruption,
        score_model_output: torch.Tensor,
        t: torch.Tensor,
        batch_idx: torch.LongTensor | None,
        batch_size: int,
        x: torch.Tensor,
        noisy_x: torch.Tensor,
        reduce: Literal["sum", "mean"],
        batch: BatchedData,
    ) -> torch.Tensor:
        """Calculate loss per sample for a single field. Returns a loss tensor of shape (batch_size,)."""
        pass


def denoising_score_matching(
    *,
    corruption: Corruption,
    score_model_output: torch.Tensor,
    t: torch.Tensor,
    batch_idx: torch.LongTensor | None,
    batch_size: int,
    x: torch.Tensor,
    noisy_x: torch.Tensor,
    reduce: Literal["sum", "mean"],
    batch: BatchedData,
    model_target: ModelTarget,
    node_is_unmasked: torch.LongTensor | None = None,
    **_,
) -> torch.Tensor:
    """Mean square error in predicting raw noise, optionally reweighted."""
    assert score_model_output.ndim >= 2
    model_target = ModelTarget(model_target)  # in case str was passed

    losses = get_losses(
        corruption=corruption,
        score_model_output=score_model_output,
        t=t,
        batch_idx=batch_idx,
        x=x,
        noisy_x=noisy_x,
        batch=batch,
        model_target=model_target,
    )

    if node_is_unmasked is not None:
        losses = node_is_unmasked.unsqueeze(-1) * losses  # Apply masking.
        original_reduce = reduce
        reduce = "sum"  # We sum first and handle the division by nodes_per_sample for the mean manually later.

    loss_per_sample = aggregate_per_sample(losses, batch_idx, reduce=reduce, batch_size=batch_size)

    if (node_is_unmasked is not None) and (original_reduce == "mean"):
        nodes_per_sample = scatter(node_is_unmasked, batch_idx, dim=0, reduce="sum")
        loss_per_sample /= nodes_per_sample

    return loss_per_sample


def get_losses(
    corruption: Corruption,
    score_model_output: torch.Tensor,
    t: torch.Tensor,
    batch_idx: torch.LongTensor | None,
    x: torch.Tensor,
    noisy_x: torch.Tensor,
    batch: BatchedData,
    model_target: ModelTarget,
) -> torch.Tensor:
    if model_target == ModelTarget.score_times_std:
        raw_noise = compute_noise_given_sample_and_corruption(
            x=x, x_noisy=noisy_x, corruption=corruption, t=t, batch_idx=batch_idx, batch=batch
        )
        target = -raw_noise
        losses = (score_model_output - target).square()
    else:
        raise ValueError(f"Unknown model_target {model_target}")
    return losses


def aggregate_per_sample(
    loss_per_row: torch.Tensor,
    batch_idx: torch.Tensor | None,
    reduce: Literal["sum", "mean"],
    batch_size: int,
):
    """
    Aggregate (potentially) batched input tensor to get a scalar for each sample in the batch.
    E.g., (num_atoms, d1, d2, ..., dn) -> (batch_size, d1, d2, ..., dn) -> (batch_size,),
    where the first aggregation only happens when batch_idx is provided.

    Args:
        loss_per_row: shape (num_nodes, any_more_dims). May contain multiple nodes per sample.
        batch_idx: shape (num_nodes,). Indicates which sample each row belongs to. If not provided,
            then we assume the first dimension is the batch dimension.
        reduce: determines how to aggregate over nodes within each sample. (Aggregation over samples
            and within dims for one node is always mean.)
        batch_size: number of samples in the batch.

    Returns:
        Scalar for each sample, shape (batch_size,).

    """
    # Sum over all but 0th dimension.
    loss_per_row = torch.mean(loss_per_row.reshape(loss_per_row.shape[0], -1), dim=1)

    if batch_idx is None:
        # First dimension is batch dimension. In this case 'reduce' is ignored.
        loss_per_sample = loss_per_row
    else:
        # Aggregate over nodes within each sample.
        loss_per_sample = scatter(
            src=loss_per_row,
            index=batch_idx,
            dim_size=batch_size,
            reduce=reduce,
        )
    return loss_per_sample


def d3pm_loss(
    *,
    corruption: Corruption,
    score_model_output: torch.Tensor,
    t: torch.Tensor,
    batch_idx: torch.LongTensor | None,
    batch_size: int,
    x: torch.Tensor,
    noisy_x: torch.Tensor,
    reduce: Literal["sum", "mean"],
    d3pm_hybrid_lambda: float = 0.0,
    **_,
) -> torch.Tensor:
    assert hasattr(corruption, "N")  # mypy
    assert hasattr(corruption, "_to_zero_based")  # mypy
    assert hasattr(corruption, "d3pm")  # mypy
    t = maybe_expand(to_discrete_time(t, N=corruption.N, T=corruption.T), batch_idx)
    metrics_dict = compute_kl_reverse_process(
        corruption._to_zero_based(x.long()),
        t,
        diffusion=corruption.d3pm,
        log_space=True,
        denoise_fn=lambda targets, timestep: score_model_output,
        hybrid_lambda=d3pm_hybrid_lambda,
        x_t_plus_1=corruption._to_zero_based(noisy_x.long()),
    )
    loss = metrics_dict.pop("loss")
    loss_per_structure = aggregate_per_sample(
        loss, batch_idx=batch_idx, reduce=reduce, batch_size=batch_size
    )
    return loss_per_structure
