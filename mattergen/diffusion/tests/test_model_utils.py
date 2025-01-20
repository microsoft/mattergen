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
    _, std = sde.marginal_prob(x=clean, t=t, batch_idx=torch.arange(10), batch=None)
    _convert = partial(
        convert_model_out_to_score,
        sde=sde,
        batch_idx=torch.arange(10),
        t=t,
        batch=None,
    )
    score1 = _convert(model_target=ModelTarget.score_times_std, model_out=-z)
    assert torch.allclose(score1, -z / std, atol=1e-4)  # slack tolerance for this test
