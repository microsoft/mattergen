# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial

import pytest
import torch

from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.model_utils import convert_model_out_to_score
from mattergen.diffusion.tests.conftest import SDE_TYPES


@pytest.mark.parametrize("sde_type", SDE_TYPES)
def test_conversions_match(sde_type):
    """Check that we get the same score whether the model output is interpreted as prediction of clean data, noise, or minus noise."""
    sde = sde_type()
    t = torch.linspace(0.1, 0.9, 10)
    clean = torch.randn(10, 3)
    z = torch.randn_like(clean)
    mean, std = sde.marginal_prob(x=clean, t=t, batch_idx=torch.arange(10), batch=None)
    noisy = mean + std * z
    _convert = partial(
        convert_model_out_to_score,
        sde=sde,
        batch_idx=torch.arange(10),
        noisy_x=noisy,
        t=t,
        batch=None,
    )
    score1 = _convert(model_target=ModelTarget.score_times_std, model_out=-z)
    score2 = _convert(model_target=ModelTarget.noise, model_out=z)
    score3 = _convert(model_target=ModelTarget.clean_data, model_out=clean)
    assert torch.allclose(score1, score2)
    assert torch.allclose(score1, score3, atol=1e-4)  # slack tolerance for this test
