# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, Type

import pytest
import torch

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.corruption.sde_lib import SDE
from mattergen.diffusion.tests.conftest import SDE_TYPES


# .sde() is only defined for continuous corruptions, hence we only test SDEs here (and not discrete corruptions)
@pytest.mark.parametrize("corruption_type", SDE_TYPES)
def test_multi_corruption(
    corruption_type: Type[SDE], tiny_state_batch, diffusion_mocks, get_multi_corruption
):
    multi_corruption = get_multi_corruption(corruption_type=corruption_type, keys=["foo", "bar"])
    t = torch.rand(tiny_state_batch.get_batch_size())

    _check_keys_shapes(multi_corruption=multi_corruption, batch=tiny_state_batch, t=t)


def _check_keys_shapes(multi_corruption: MultiCorruption, batch, t: torch.Tensor):
    drifts_diffusions = multi_corruption.sde(batch=batch, t=t)
    _assert_keys(drifts_diffusions)

    for k, (drift, diffusion) in drifts_diffusions.items():
        assert drift.shape == batch[k].shape
        assert diffusion.shape[0] == batch[k].shape[0]


def _assert_keys(d: Dict[str, Any]):
    assert set(d.keys()) == {"foo", "bar"}
