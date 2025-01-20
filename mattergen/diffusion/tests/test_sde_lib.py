# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Type

import pytest
import torch

from mattergen.diffusion.corruption.sde_lib import SDE
from mattergen.diffusion.tests.conftest import SDE_TYPES


def _check_batch_shape(x: torch.Tensor, batch_size: torch.LongTensor):
    """Checks sde outputs that should be (batch_size, )"""
    assert len(x.shape) == 1
    assert x.shape[0] == batch_size


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("sdetype", SDE_TYPES)
def test_sde(tiny_state_batch, sdetype: Type[SDE], sparse, EPS):
    """Tests correct shapes for all methods of the SDE class"""
    x: torch.Tensor = tiny_state_batch["foo"]
    sde: SDE = sdetype()

    if sparse:
        batch_size = tiny_state_batch.get_batch_size()
        batch_idx = tiny_state_batch.get_batch_idx("foo")
    else:
        batch_size = x.shape[0]
        batch_idx = None

    t = torch.rand(batch_size) * (sde.T - EPS) + EPS

    def _check_shapes(drift, diffusion):
        assert drift.shape == x.shape
        assert diffusion.shape[0] == x.shape[0]

    # Forward SDE methods
    drift, diffusion = sde.sde(x, t, batch_idx)
    _check_shapes(drift, diffusion)

    mean, std = sde.marginal_prob(x, t, batch_idx)

    _check_shapes(mean, std)

    z = sde.prior_sampling(x.shape)

    assert z.shape == x.shape

    prior_logp = sde.prior_logp(z, batch_idx=batch_idx)

    _check_batch_shape(prior_logp, batch_size)


def dummy_score_fn(x, t, batch_idx):
    return torch.zeros_like(x)
