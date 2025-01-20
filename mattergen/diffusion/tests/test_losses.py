# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Dict, List, Type

import pytest
import torch

from mattergen.diffusion.corruption.corruption import Corruption
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.corruption.sde_lib import SDE
from mattergen.diffusion.data.batched_data import SimpleBatchedData
from mattergen.diffusion.losses import DenoisingScoreMatchingLoss
from mattergen.diffusion.tests.conftest import SDE_TYPES
from mattergen.diffusion.training.field_loss import (
    aggregate_per_sample,
    compute_noise_given_sample_and_corruption,
)
from mattergen.diffusion.wrapped.wrapped_normal_loss import wrapped_normal_loss
from mattergen.diffusion.wrapped.wrapped_sde import WrappedVESDE


def get_multi_corruption(corruption_type, keys: List[str]):
    discrete_corruptions = {
        k: corruption_type()
        for k in keys
        if issubclass(corruption_type, Corruption) and not issubclass(corruption_type, SDE)
    }
    sdes = {k: corruption_type() for k in keys if issubclass(corruption_type, SDE)}
    return MultiCorruption(sdes=sdes, discrete_corruptions=discrete_corruptions)


@pytest.mark.parametrize("corruption_type", SDE_TYPES)
def test_calc_loss(tiny_state_batch, corruption_type: Type[Corruption]):
    """Check that calc_loss returns expected values for a few examples."""

    clean_batch = tiny_state_batch
    multi_corruption = get_multi_corruption(corruption_type=corruption_type, keys=["foo", "bar"])

    t = torch.ones(clean_batch.get_batch_size())
    noisy_batch = multi_corruption.sample_marginal(batch=clean_batch, t=t)

    raw_noise = apply(
        {k: compute_noise_given_sample_and_corruption for k in multi_corruption.corrupted_fields},
        x=clean_batch,
        x_noisy=noisy_batch,
        corruption=multi_corruption.corruptions,
        batch_idx=clean_batch.batch_idx,
        broadcast={"t": t, "batch": clean_batch},
    )

    zero_scores = {k: torch.zeros_like(v) for k, v in clean_batch.data.items()}
    calc_loss = partial(
        DenoisingScoreMatchingLoss(
            model_targets={"foo": "score_times_std"},
        ),
        multi_corruption=multi_corruption,
        t=t,
        batch=clean_batch,
    )

    score_model_output = SimpleBatchedData(data=zero_scores, batch_idx=clean_batch.batch_idx)
    loss, _ = calc_loss(score_model_output=score_model_output, noisy_batch=noisy_batch)
    target_loss = aggregate_per_sample(
        raw_noise["foo"].pow(2),
        batch_idx=clean_batch.batch_idx["foo"],
        reduce="mean",
        batch_size=clean_batch.get_batch_size(),
    ).mean()
    torch.testing.assert_allclose(loss, target_loss)

    # Errors in bar should not affect the loss, only foo.
    score_model_output = score_model_output.replace(bar=score_model_output["bar"] + 100)

    loss_with_bad_bar, _ = calc_loss(score_model_output=score_model_output, noisy_batch=noisy_batch)

    torch.testing.assert_allclose(loss, loss_with_bad_bar)

    # Increasing error in foo should increase the loss; doubling raw noise leads to 4x loss.
    raw_noise.update(foo=raw_noise["foo"] * 2)
    mean, std = multi_corruption.corruptions["foo"].marginal_prob(
        x=clean_batch["foo"],
        t=t[clean_batch.batch_idx["foo"]],
        batch_idx=clean_batch.batch_idx["foo"],
        batch=clean_batch,
    )
    noisy_batch = clean_batch.replace(foo=raw_noise["foo"] * std + mean)
    loss, _ = calc_loss(score_model_output=score_model_output, noisy_batch=noisy_batch)
    torch.testing.assert_allclose(
        loss,
        target_loss * 4,
    )


@pytest.mark.parametrize("corruption_type", SDE_TYPES)
def test_weighted_summed_field_loss(
    tiny_state_batch,
    corruption_type: Type[Corruption],
):
    """Check that SummedFieldLoss returns expected values for a few examples."""

    clean_batch = tiny_state_batch
    multi_corruption = get_multi_corruption(
        corruption_type=corruption_type,
        keys=[
            "foo",
            "bar",
        ],
    )
    zero_scores = {k: torch.zeros_like(v) for k, v in clean_batch.data.items()}
    score_model_output = SimpleBatchedData(data=zero_scores, batch_idx=clean_batch.batch_idx)
    t = torch.ones(clean_batch.get_batch_size())
    noisy_batch = multi_corruption.sample_marginal(batch=clean_batch, t=t)

    weights = {
        "foo": 1.0,
        "bar": 2.9,
    }
    model_targets: Dict[str, str] = {
        k: "score_times_std" for k in multi_corruption.corrupted_fields
    }
    unweighted_loss_fn = DenoisingScoreMatchingLoss(model_targets=model_targets)
    weighted_loss_fn = DenoisingScoreMatchingLoss(
        weights=weights,
        model_targets=model_targets,
    )

    unweighted_loss, unweighted_loss_per_field = unweighted_loss_fn(
        batch=clean_batch,
        multi_corruption=multi_corruption,
        t=t,
        score_model_output=score_model_output,
        noisy_batch=noisy_batch,
    )
    weighted_loss, weighted_loss_per_field = weighted_loss_fn(
        batch=clean_batch,
        multi_corruption=multi_corruption,
        t=t,
        score_model_output=score_model_output,
        noisy_batch=noisy_batch,
    )
    torch.testing.assert_allclose(
        weighted_loss,
        unweighted_loss_per_field["foo"] * weights["foo"]
        + unweighted_loss_per_field["bar"] * weights["bar"],
    )
    torch.testing.assert_allclose(
        torch.stack([unweighted_loss_per_field[k] for k in unweighted_loss_per_field.keys()]),
        torch.stack([weighted_loss_per_field[k] for k in weighted_loss_per_field.keys()]),
    )
    torch.testing.assert_allclose(sum(weighted_loss_per_field.values()), unweighted_loss)


def test_wrapped_normal_loss(tiny_state_batch):
    # Simulate the case that wrapping has basically no effect and the loss is equivalent to DenoisingScoreMatchingLoss
    clean_batch = tiny_state_batch.replace(
        foo=tiny_state_batch["foo"] + 500, bar=tiny_state_batch["bar"][:, :3] + 500
    )
    fields = ["foo", "bar"]
    multi_corruption: MultiCorruption = MultiCorruption(
        sdes={k: WrappedVESDE(wrapping_boundary=1000.0, sigma_max=1.0) for k in fields}
    )
    model_targets = {k: "score_times_std" for k in fields}
    zero_scores = {k: torch.zeros_like(v) for k, v in clean_batch.data.items()}
    score_model_output = SimpleBatchedData(data=zero_scores, batch_idx=clean_batch.batch_idx)
    t = torch.rand(clean_batch.get_batch_size())
    noisy_batch = multi_corruption.sample_marginal(batch=clean_batch, t=t)
    wrapped_loss_foo = wrapped_normal_loss(
        corruption=multi_corruption.sdes["foo"],
        score_model_output=score_model_output["foo"],
        t=t,
        batch_idx=clean_batch.get_batch_idx("foo"),
        batch_size=clean_batch.get_batch_size(),
        x=clean_batch["foo"],
        noisy_x=noisy_batch["foo"],
        batch=clean_batch,
        reduce="mean",
    ).mean()
    wrapped_loss_bar = wrapped_normal_loss(
        corruption=multi_corruption.sdes["bar"],
        score_model_output=score_model_output["bar"],
        t=t,
        batch_idx=clean_batch.get_batch_idx("bar"),
        batch_size=clean_batch.get_batch_size(),
        x=clean_batch["bar"],
        noisy_x=noisy_batch["bar"],
        batch=clean_batch,
        reduce="mean",
    ).mean()
    wrapped_loss = {"foo": wrapped_loss_foo, "bar": wrapped_loss_bar}
    non_wrapped_loss_fn = DenoisingScoreMatchingLoss(
        model_targets=model_targets,
    )
    _, non_wrapped_loss_per_field = non_wrapped_loss_fn(
        batch=clean_batch,
        multi_corruption=multi_corruption,
        t=t,
        score_model_output=score_model_output,
        noisy_batch=noisy_batch,
    )
    torch.testing.assert_allclose(
        torch.stack([wrapped_loss[k] for k in wrapped_loss.keys()]),
        torch.stack([non_wrapped_loss_per_field[k] for k in non_wrapped_loss_per_field.keys()]),
    )
