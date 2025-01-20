# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This is an integeratation test of reverse sampling. For a known data distribution that
is Gaussian, we substitute the known ground truth score for an approximate model
prediction and reverse sample to check we retrieve correct moments of the data distribution.
"""

from argparse import Namespace
from contextlib import nullcontext
from functools import partial
from typing import List, Type

import pytest
import torch

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.corruption.sde_lib import SDE
from mattergen.diffusion.data.batched_data import BatchedData, SimpleBatchedData
from mattergen.diffusion.diffusion_module import DiffusionModule
from mattergen.diffusion.exceptions import IncompatibleSampler
from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector
from mattergen.diffusion.tests.conftest import (
    DEFAULT_CORRECTORS,
    DEFAULT_PREDICTORS,
    SDE_TYPES,
    WRAPPED_CORRECTORS,
    WRAPPED_PREDICTORS,
)
from mattergen.diffusion.tests.test_sampling import INCOMPATIBLE_SAMPLERS
from mattergen.diffusion.wrapped.wrapped_sde import WrappedVESDE, WrappedVPSDE


def score_given_xt(
    x: BatchedData,
    t: torch.Tensor,
    multi_corruption: MultiCorruption,
    x0_mean: torch.Tensor,
    x0_std: torch.Tensor,
) -> BatchedData:
    def _score_times_std(x_t: torch.Tensor, sde: SDE) -> torch.Tensor:
        a_t, s_t = sde.marginal_prob(x=torch.ones_like(x_t), t=t)
        mean = a_t * x0_mean
        std = torch.sqrt(a_t**2 * x0_std**2 + s_t**2)
        score_times_std = -(x_t - mean) / (std**2) * s_t
        return score_times_std

    return x.replace(
        **{
            k: _score_times_std(x_t=x[k], sde=multi_corruption.sdes[k])
            for k in multi_corruption.sdes.keys()
        }
    )


def get_diffusion_module(x0_mean, x0_std, multi_corruption: MultiCorruption) -> DiffusionModule:
    return DiffusionModule(
        model=partial(score_given_xt, x0_mean=x0_mean, x0_std=x0_std, multi_corruption=multi_corruption),  # type: ignore
        corruption=multi_corruption,
        loss_fn=Namespace(
            model_targets={k: ModelTarget.score_times_std for k in multi_corruption.sdes.keys()}
        ),  # type: ignore
    )


predictor_corrector_pairs = [(p, None) for p in DEFAULT_PREDICTORS] + [
    (None, c) for c in DEFAULT_CORRECTORS
]


@pytest.mark.parametrize(
    "predictor_type,corrector_type",
    predictor_corrector_pairs,
)
@pytest.mark.parametrize("corruption_type", SDE_TYPES)
def test_reverse_sampling(corruption_type: Type, predictor_type: Type, corrector_type: Type):
    N = 1000 if corrector_type is None else 200

    if predictor_type is None and corrector_type is None:
        # Nothing to be done here.
        return
    fields = ["x", "y", "z", "a"]
    batch_size = 10_000
    x0_mean = torch.tensor(-3.0)
    x0_std = torch.tensor(4.3)

    multi_corruption: MultiCorruption = MultiCorruption(sdes={f: corruption_type() for f in fields})

    with pytest.raises(IncompatibleSampler) if predictor_type in INCOMPATIBLE_SAMPLERS[
        corruption_type
    ] or corrector_type in INCOMPATIBLE_SAMPLERS[corruption_type] else nullcontext():
        multi_sampler = PredictorCorrector(
            diffusion_module=get_diffusion_module(
                multi_corruption=multi_corruption, x0_mean=x0_mean, x0_std=x0_std
            ),
            device=torch.device("cpu"),
            predictor_partials={} if predictor_type is None else {k: predictor_type for k in fields},  # type: ignore
            corrector_partials={} if corrector_type is None else {k: corrector_type for k in fields},  # type: ignore
            n_steps_corrector=5,
            N=N,
            eps_t=0.001,
            max_t=None,
        )
        conditioning_data = _get_conditioning_data(batch_size=batch_size, fields=fields)

        samples, _ = multi_sampler.sample(conditioning_data=conditioning_data)
        means = torch.tensor([samples[k].mean() for k in multi_corruption.corruptions.keys()])
        stds = torch.tensor([samples[k].std() for k in multi_corruption.corruptions.keys()])
        assert torch.isclose(means.mean(), x0_mean, atol=1e-1)
        assert torch.isclose(stds.mean(), x0_std, atol=1e-1)


wrapped_pc_pairs = [(p, None) for p in WRAPPED_PREDICTORS] + [(None, c) for c in WRAPPED_CORRECTORS]


@pytest.mark.parametrize("predictor_type, corrector_type", wrapped_pc_pairs)
@pytest.mark.parametrize("sde_type", [WrappedVESDE, WrappedVPSDE])
def test_wrapped_reverse_sampling(sde_type: Type, predictor_type: Type, corrector_type: Type):
    if predictor_type is None and corrector_type is None:
        # Nothing to be done here.
        return
    N = 50
    fields = ["x", "y", "z", "a"]
    batch_size = 10_000
    x0_mean = torch.tensor(-2.0)
    x0_std = torch.tensor(2.3)
    wrapping_boundary = -2.4
    empirical_samples = torch.remainder(
        torch.randn(batch_size) * x0_std + x0_mean, wrapping_boundary
    )
    empirical_x0_mean = empirical_samples.mean()
    empirical_x0_std = empirical_samples.std()

    multi_corruption: MultiCorruption = MultiCorruption(
        sdes={k: sde_type(wrapping_boundary=wrapping_boundary) for k in fields}
    )

    predictor_partials = {} if predictor_type is None else {k: predictor_type for k in fields}
    corrector_partials = {} if corrector_type is None else {k: corrector_type for k in fields}

    n_steps_corrector = 5

    multi_sampler: PredictorCorrector = PredictorCorrector(
        diffusion_module=get_diffusion_module(
            x0_mean=x0_mean, x0_std=x0_std, multi_corruption=multi_corruption
        ),
        n_steps_corrector=n_steps_corrector,
        predictor_partials=predictor_partials,  # type: ignore
        corrector_partials=corrector_partials,  # type: ignore
        device=None,
        N=N,
    )

    conditioning_data = _get_conditioning_data(batch_size=batch_size, fields=fields)
    (samples, _) = multi_sampler.sample(conditioning_data=conditioning_data, mask=None)
    assert min(samples[k].min() for k in multi_corruption.corruptions.keys()) >= wrapping_boundary
    assert max(samples[k].max() for k in multi_corruption.corruptions.keys()) <= 0.0
    means = torch.tensor([samples[k].mean() for k in multi_corruption.corruptions.keys()])
    stds = torch.tensor([samples[k].std() for k in multi_corruption.corruptions.keys()])
    assert torch.isclose(means.mean(), empirical_x0_mean, atol=1e-1)
    assert torch.isclose(stds.mean(), empirical_x0_std, atol=1e-1)


def _get_conditioning_data(batch_size: int, fields: List[str]) -> SimpleBatchedData:
    return SimpleBatchedData(
        data={k: torch.randn(batch_size, 1) for k in fields}, batch_idx={k: None for k in fields}
    )
