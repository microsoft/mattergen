# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
from contextlib import nullcontext
from typing import Callable, Dict, List, Type, Union

import pytest
import torch

from mattergen.diffusion.corruption.sde_lib import SDE, VESDE, VPSDE
from mattergen.diffusion.d3pm.d3pm_predictors_correctors import D3PMAncestralSamplingPredictor
from mattergen.diffusion.exceptions import IncompatibleSampler
from mattergen.diffusion.sampling import predictors_correctors as pc
from mattergen.diffusion.sampling.predictors import AncestralSamplingPredictor, Predictor
from mattergen.diffusion.tests.conftest import (
    DEFAULT_CORRECTORS,
    DEFAULT_PREDICTORS,
    SDE_TYPES,
    WRAPPED_CORRECTORS,
    WRAPPED_PREDICTORS,
)
from mattergen.diffusion.wrapped.wrapped_predictors_correctors import (
    WrappedAncestralSamplingPredictor,
    WrappedLangevinCorrector,
)
from mattergen.diffusion.wrapped.wrapped_sde import WrappedVESDE, WrappedVPSDE

D3PM_SAMPLERS = [
    D3PMAncestralSamplingPredictor,
]
INCOMPATIBLE_SAMPLERS: Dict[
    Type[SDE], List[Type[Union[Predictor, pc.LangevinCorrector]]]
] = defaultdict(list)
INCOMPATIBLE_SAMPLERS[VPSDE] = [
    WrappedLangevinCorrector,
    WrappedAncestralSamplingPredictor,
    *D3PM_SAMPLERS,
]
INCOMPATIBLE_SAMPLERS[VESDE] = [
    WrappedLangevinCorrector,
    WrappedAncestralSamplingPredictor,
    *D3PM_SAMPLERS,
]
INCOMPATIBLE_SAMPLERS[WrappedVPSDE] = [
    AncestralSamplingPredictor,
    pc.LangevinCorrector,
    *D3PM_SAMPLERS,
]
INCOMPATIBLE_SAMPLERS[WrappedVESDE] = [
    AncestralSamplingPredictor,
    pc.LangevinCorrector,
    *D3PM_SAMPLERS,
]


@pytest.mark.parametrize("predictor_type", DEFAULT_PREDICTORS + WRAPPED_PREDICTORS)
@pytest.mark.parametrize("sde_type", SDE_TYPES)
def test_predictor(make_state_batch: Callable, predictor_type: Type, sde_type, EPS: float):
    """Tests whether implemented predictors return arrays of consistent
    graph shape
    """
    tiny_state_batch = make_state_batch(sde_type)

    with pytest.raises(IncompatibleSampler) if predictor_type in INCOMPATIBLE_SAMPLERS[
        sde_type
    ] else nullcontext():
        sde = sde_type()
        batch_size = tiny_state_batch.get_batch_size()
        t = torch.rand(batch_size) * (sde.T - EPS) + EPS

        old_x: torch.Tensor = tiny_state_batch["foo"]

        pr: Predictor = predictor_type(corruption=sde, score_fn=dummy_score_fn)
        dt = torch.tensor(-(sde.T - EPS) / 50)
        x, x_mean = pr.update_fn(
            x=old_x,
            t=t,
            dt=dt,
            batch_idx=tiny_state_batch.get_batch_idx("foo"),
            batch=tiny_state_batch,
        )

        assert x.shape == x_mean.shape == old_x.shape


def dummy_score_fn(x, t, batch_idx):
    score = torch.zeros(*x.shape[:2])
    return score


@pytest.mark.parametrize("corrector_type", DEFAULT_CORRECTORS + WRAPPED_CORRECTORS)
@pytest.mark.parametrize("sde_type", SDE_TYPES)
def test_corrector(make_state_batch: Callable, corrector_type: Type, sde_type, EPS: float):
    """Tests whether implemented correctors return arrays of consistent
    graph shape
    """
    tiny_state_batch = make_state_batch(sde_type)

    with pytest.raises(IncompatibleSampler) if corrector_type in INCOMPATIBLE_SAMPLERS[
        sde_type
    ] else nullcontext():
        sde = sde_type()
        t = torch.rand(tiny_state_batch.get_batch_size()) * (sde.T - EPS) + EPS
        old_x: torch.Tensor = tiny_state_batch["foo"]

        corrector: pc.LangevinCorrector = corrector_type(sde, score_fn=dummy_score_fn, n_steps=5)

        x, x_mean = corrector.update_fn(
            x=old_x, t=t, batch_idx=tiny_state_batch.get_batch_idx("foo")
        )

        assert x.shape == x_mean.shape == old_x.shape
