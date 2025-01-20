# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from math import pi
from typing import Optional, Tuple

import pytest
import torch

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.diffusion.corruption import (
    LatticeVPSDE,
    expand,
    make_noise_symmetric_preserve_variance,
)
from mattergen.diffusion.corruption.sde_lib import VPSDE


class TestVPSDE(VPSDE, ABC):
    @classmethod
    @abstractmethod
    def get_random_data(cls, N: int) -> torch.Tensor:
        pass

    @abstractmethod
    def get_limit_mean(
        self, x: torch.Tensor, limit_info: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def get_limit_var(
        self, x: torch.Tensor, limit_info: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def assert_discretize_ok(self, x: torch.Tensor) -> None:
        pass


def test_LatticeVPSDE_get_limit_mean():
    density = 15.0  # (atoms/Angstrom**3) - this is a very large magnitude to ensure signal>>noise in this unit test

    sde = LatticeVPSDE(limit_density=density, limit_mean="scaled")

    # number atoms per crystal
    n_atoms = torch.tensor([1, 2])
    batch = ChemGraph(num_atoms=n_atoms)

    # crystal lattices are not used in LatticeVPSDE.get_limit_mean, shape=[2, 3, 3]
    lattices = torch.eye(3).expand(2, 3, 3)

    # shape=[2, 3, 3]
    lattice_mean = sde.get_limit_mean(x=lattices, batch=batch)

    # expected value on diagonals is (n_atoms/density)**(1/3)
    expected_val = torch.pow(n_atoms / density, 1 / 3)

    assert torch.allclose(lattice_mean[0], expected_val[0] * torch.eye(3))
    assert torch.allclose(lattice_mean[1], expected_val[1] * torch.eye(3))


def test_LatticeVPSDE_get_var_mean():
    density = 20.0  # (atoms/Angstrom**3)

    sde = LatticeVPSDE(limit_density=density)

    # number atoms per crystal
    n_atoms = torch.tensor([1, 2])
    batch = ChemGraph(num_atoms=n_atoms)
    # crystal lattices are not used in LatticeVPSDE.get_limit_var, shape=[2, 3, 3]
    lattices = torch.eye(3).expand(2, 3, 3)

    # shape=[2, 3, 3]
    lattice_var = sde.get_limit_var(x=lattices, batch=batch)

    # expected variance is n_atoms**(2/3), shape=(2, 3, 3)
    expected_val = (
        expand(torch.pow(n_atoms, 2 / 3), (2, 3, 3)).tile(1, 3, 3) * sde.limit_var_scaling_constant
    )

    assert torch.allclose(lattice_var, expected_val)


def test_LatticeVPSDE_prior_sampling():
    # limit density (atoms/Angstrom**3)
    density = 20.0
    # number crystals
    Nbatch = 1000
    # 10 atoms per crystal
    n_atoms = torch.ones((Nbatch,)) * 10
    batch = ChemGraph(num_atoms=n_atoms)

    sde = LatticeVPSDE(limit_density=density)

    # sample from noisy limit prior, all elements are ~N(0, 1)
    x = sde.prior_sampling(shape=(Nbatch, 3, 3), conditioning_data=batch)

    expected_mean = sde.get_limit_mean(x=x, batch=batch).mean(0)
    expected_var = sde.get_limit_var(x=x, batch=batch).mean(0)[0, 0]

    assert x.shape == (Nbatch, 3, 3)
    # all elements in noisy state should be IID as N(0,1)
    assert torch.allclose(x.mean(0), expected_mean, atol=1e-1)
    assert torch.allclose(x.var(0).mean(), expected_var, atol=1e-1)


def test_LatticeVPSDE_prior_logp():
    # evaluate the log likelihood of sample x~p_T the noisy limit distribution

    # limit density (atoms/Angstrom**3)
    density = 20.0
    # number crystals
    Nbatch = 100
    # 10 atoms per crystal
    n_atoms = torch.ones((Nbatch,)) * 10
    batch = ChemGraph(num_atoms=n_atoms)

    sde = LatticeVPSDE(limit_density=density, limit_var_scaling_constant=1.0)

    # sample from noisy limit prior, all elements are ~N(0, 1)
    x = sde.prior_sampling(shape=(Nbatch, 3, 3), conditioning_data=batch)

    # pdf for standard normal = exp(-x**2/2)/sqrt(2 pi ), shape=(Nbatch, 3, 3)
    expected_log_likelihood = -0.5 * torch.pow(x, 2) - 0.5 * torch.log(torch.tensor([2.0 * pi]))

    # sum over IID contributions from data dimensions
    expected_log_likelihood = torch.sum(expected_log_likelihood, dim=(-2, -1))

    assert torch.allclose(sde.prior_logp(z=x, batch=batch), expected_log_likelihood)


def test_LatticeVPSDE_marginal_prob():
    # check mean and standard deviation of the distribution p(x_t|x_0)

    # limit density (atoms/Angstrom**3)
    density = 20.0
    # number crystals
    Nbatch = 100
    # 10 atoms per crystal
    n_atoms = torch.ones((Nbatch,)) * 10
    batch = ChemGraph(num_atoms=n_atoms)

    sde = LatticeVPSDE(limit_density=density, limit_var_scaling_constant=1.0)

    t = torch.ones((1,)) * 0.5
    x = torch.ones(Nbatch, 3, 3)

    # get moments for p(x_t | x_0)
    mean, std = sde.marginal_prob(x=x, t=t, batch=batch)

    # p(x_t|x_0) = N(x_t | mean, var) as per Eq. 33 in https://arxiv.org/pdf/2011.13456v2.pdf
    coeff = torch.exp(-0.25 * (t**2) * (sde.beta_1 - sde.beta_0) - 0.5 * t * sde.beta_0)
    expected_mean = coeff * x + (1 - coeff)[:, None, None] * (
        torch.eye(3)[None] * batch.num_atoms[:, None, None] / density
    ).pow(1.0 / 3)

    # vanilla term for unit variance noisy limit distribution
    expected_var = 1 - torch.exp(-0.5 * (t**2) * (sde.beta_1 - sde.beta_0) - t * sde.beta_0)

    # account for fact we have non unit variance limit distribution in general
    expected_var = expected_var * sde.get_limit_var(x=x, batch=batch)

    assert mean.shape == (Nbatch, 3, 3)
    assert std.shape == (Nbatch, 3, 3)
    assert torch.allclose(expected_mean, mean)
    assert torch.allclose(expected_var.sqrt(), std)


def test_make_noise_symmetric_preserve_variance():
    noise = torch.randn(100_000, 3, 3)
    symmetric_noise = make_noise_symmetric_preserve_variance(noise)
    assert torch.allclose(noise.var(), symmetric_noise.var(), atol=1e-2)
    assert torch.allclose(noise.mean(), symmetric_noise.mean(), atol=1e-2)

    # should raise an assertion error if noise is not a (batched) square matrix
    with pytest.raises(AssertionError):
        make_noise_symmetric_preserve_variance(torch.randn(100_000, 3, 4))
    with pytest.raises(AssertionError):
        make_noise_symmetric_preserve_variance(
            torch.randn(
                100_000,
                3,
            )
        )
    with pytest.raises(AssertionError):
        make_noise_symmetric_preserve_variance(torch.randn(100_000, 3, 1))


@pytest.mark.parametrize("output_shape", [(10, 3, 3), (10, 3, 1), (10, 3), (10, 2), (10, 3, 9, 1)])
def test_expand(output_shape: Tuple):
    unexpanded_data = torch.randn((10,))
    expanded_data = expand(unexpanded_data, output_shape)

    assert len(expanded_data.shape) == len(output_shape)

    # we only match the len, not number of elements
    assert expanded_data.shape != output_shape
