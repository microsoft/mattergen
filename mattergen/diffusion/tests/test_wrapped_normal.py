# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from mattergen.diffusion.wrapped.wrapped_normal_loss import get_pbc_offsets, wrapped_normal_score
from mattergen.diffusion.wrapped.wrapped_sde import wrap_at_boundary


def test_wrapped_normal_score_isotropic():
    variance = torch.rand((1,)) * 5
    max_offsets = 3
    num_atoms = 1
    batch = torch.zeros(num_atoms, dtype=torch.long)
    cell = torch.tensor([[[3.4641, 0.0, 2.0], [-1.1196, 1.6572, 0], [0.0, 0.0, 3.0]]])
    lattice_offsets = get_pbc_offsets(cell, max_offsets)

    mean = torch.zeros((3,))
    shifted_means = mean[None, None] + lattice_offsets

    normal_distributions = Normal(shifted_means[0], variance.sqrt().item())
    noisy_frac_coords = torch.rand((num_atoms, 3))
    noisy_cart_coords = wrap_at_boundary(noisy_frac_coords, 1.0)
    noisy_cart_coords.requires_grad = True
    comp_scores = wrapped_normal_score(
        noisy_cart_coords,
        mean[None],
        cell,
        variance.repeat(num_atoms),
        batch,
        max_offsets,
    )

    mix = Categorical(probs=torch.ones(shifted_means.shape[1]))
    comp = Independent(normal_distributions, 1)
    gmm = MixtureSameFamily(mix, comp)
    gmm.log_prob(noisy_cart_coords).backward()
    coord_score = noisy_cart_coords.grad
    assert torch.allclose(coord_score, comp_scores, atol=1e-5)
