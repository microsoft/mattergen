# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from omegaconf import DictConfig

from mattergen.diffusion.corruption.corruption import B, BatchedData, maybe_expand
from mattergen.diffusion.corruption.sde_lib import SDE as DiffSDE
from mattergen.diffusion.corruption.sde_lib import VESDE as DiffVESDE
from mattergen.diffusion.corruption.sde_lib import VPSDE
from mattergen.diffusion.wrapped.wrapped_sde import WrappedVESDE


def expand(a, x_shape, left=False):
    a_dim = len(a.shape)
    if left:
        return a.reshape(*(((1,) * (len(x_shape) - a_dim)) + a.shape))
    else:
        return a.reshape(*(a.shape + ((1,) * (len(x_shape) - a_dim))))


def make_noise_symmetric_preserve_variance(noise: torch.Tensor) -> torch.Tensor:
    """Makes the noise matrix symmetric, preserving the variance. Assumes i.i.d. noise for each dimension.

    Args:
        noise (torch.Tensor): Input noise matrix, must be a batched square matrix, i.e., have shape (batch_size, dim, dim).

    Returns:
        torch.Tensor: The symmetric noise matrix, with the same variance as the input.
    """
    assert (
        len(noise.shape) == 3 and noise.shape[1] == noise.shape[2]
    ), "Symmetric noise only works for square-matrix-shaped data."
    # Var[1/sqrt(2) * (eps_i + eps_j)] = 0.5 Var[eps_i] + 0.5 Var[eps_j] = Var[noise]
    # Special treatment of the diagonal elements, i.e., those we leave unchanged via masking.
    return (1 / (2**0.5)) * (1 - torch.eye(3, device=noise.device)[None]) * (
        noise + noise.transpose(1, 2)
    ) + torch.eye(3, device=noise.device)[None] * noise


class LatticeVPSDE(VPSDE):
    @staticmethod
    def from_vpsde_config(vpsde_config: DictConfig):
        return LatticeVPSDE(
            **vpsde_config,
        )

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20,
        limit_density: float | None = 0.05,
        limit_var_scaling_constant: float = 0.25,
        **kwargs,
    ):
        """Variance-preserving SDE with drift coefficient changing linearly over time."""
        super().__init__()
        self.beta_0 = beta_min
        self.beta_1 = beta_max

        # each crystal is diffused to have expected lattice vectors
        # based on the number of atoms per crystal and self.limit_density
        # units=(atoms/Angstrom**3)
        self.limit_density = limit_density
        self.limit_var_scaling_constant = limit_var_scaling_constant

        self._limit_info_key = "num_atoms"

    @property
    def limit_info_key(self) -> str:
        return self._limit_info_key

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def _marginal_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        return torch.exp(log_mean_coeff)

    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert batch is not None
        mean_coeff = self._marginal_mean_coeff(t)
        # x: shape [batch_size, *x.shape[1:]]
        # t, limit_info: shape [batch_size,]
        limit_mean = self.get_limit_mean(x=x, batch=batch)
        limit_var = self.get_limit_var(x=x, batch=batch)
        mean_coeff_expanded = maybe_expand(mean_coeff, batch_idx, x)
        mean = mean_coeff_expanded * x + (1 - mean_coeff_expanded) * limit_mean
        std = torch.sqrt((1.0 - mean_coeff_expanded**2) * limit_var)
        return mean, std

    def mean_coeff_and_std(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns mean coefficient and standard deviation of marginal distribution at time t."""
        mean_coeff = self._marginal_mean_coeff(t)
        std = self.marginal_prob(x, t, batch_idx, batch)[1]
        return maybe_expand(mean_coeff, batch=None, like=x), std

    def get_limit_mean(self, x: torch.Tensor, batch: BatchedData) -> torch.Tensor:
        # x: shape [batch_size, *x.shape[1:]]
        # limit_info: shape [batch_size,], a 1d tensor containing number of atoms per crystal
        # self.limit_density = limit_info / mean lattice vector length**3

        # shape=[Ncrystals,]
        n_atoms = batch[self.limit_info_key]

        # shape=[Ncrystals, 3, 3]
        return torch.pow(
            torch.eye(3, device=x.device).expand(len(n_atoms), 3, 3)
            * n_atoms[:, None, None]
            / self.limit_density,
            1.0 / 3,
        ).to(x.device)

    def get_limit_var(self, x: torch.Tensor, batch: BatchedData) -> torch.Tensor:
        """
        Returns the element-wise variance of the limit distribution.
        NOTE: even though we have a different limit variance per data
        dimension we still sample IID for each element per data point.
        We do NOT do any correlated sampling over data dimensions per
        data point.

        Return shape=x.shape
        """

        # x: shape [batch_size, *x.shape[1:]]
        # limit_info: shape [batch_size,]
        # necessary for mypy
        n_atoms = batch[self.limit_info_key]

        # expand to fit shape of data, shape = (n_crystals, 1, 1)
        n_atoms_expanded = expand(n_atoms, x.shape)

        # shape = (n_crystals, 3, 3)
        n_atoms_expanded = torch.tile(n_atoms_expanded, (1, 3, 3))

        # scale limit standard deviation to be proportional to number atoms = n_atoms**(1/3)
        # per lattice vector. We hope that prod_i std_i scales as the standard deviation
        # of the actual volume. NOTE: we return variance here, hence 2 in the power
        # shape=(Ncrystals, 3, 3) for limit_info.shape=[Ncrystals,]
        out = torch.pow(n_atoms_expanded, 2.0 / 3).to(x.device) * self.limit_var_scaling_constant

        return out

    def sample_marginal(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> torch.Tensor:
        mean, std = self.marginal_prob(x=x, t=t, batch=batch)
        z = torch.randn_like(x)
        z = make_noise_symmetric_preserve_variance(z)
        return mean + expand(std, z.shape) * z

    def prior_sampling(
        self,
        shape: torch.Size | tuple,
        conditioning_data: BatchedData | None = None,
        batch_idx: B = None,
    ) -> torch.Tensor:
        x_sample = torch.randn(*shape)
        x_sample = make_noise_symmetric_preserve_variance(x_sample)
        assert conditioning_data is not None
        limit_info = conditioning_data[self.limit_info_key]
        x_sample = x_sample.to(limit_info.device)
        limit_mean = self.get_limit_mean(x=x_sample, batch=conditioning_data)
        limit_var = self.get_limit_var(x=x_sample, batch=conditioning_data)
        # shape=[Nbatch,...] for shape[0]=Nbatch
        return x_sample * limit_var.sqrt() + limit_mean

    def sde(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert batch is not None
        # x: shape [batch_size, *x.shape[1:]]
        # t, limit_info: shape [batch_size,]
        # if a per data-point limit mean is supplied, expand to shape of data
        # shape=x.shape
        limit_mean = self.get_limit_mean(x=x, batch=batch)

        # if a per data-point limit variance is supplied, shape=[x.shape[0], ]
        limit_var = self.get_limit_var(x=x, batch=batch)

        beta_t = self.beta(t)
        drift = (
            -0.5
            * expand(
                beta_t,
                x.shape,
            )
            * (x - limit_mean)
        )
        diffusion = torch.sqrt(expand(beta_t, limit_var.shape) * limit_var)
        # drift.shape=[Nbatch,...], diffusion.shape=[Nbatch,] for x.shape[0]=Nbatch
        return maybe_expand(drift, batch_idx), maybe_expand(diffusion, batch_idx)


class NumAtomsVarianceAdjustedWrappedVESDE(WrappedVESDE):
    """Wrapped VESDE with variance adjusted by number of atoms. We divide the standard deviation by the cubic root of the number of atoms.
    The goal is to reduce the influence by the cell size on the variance of the fractional coordinates.
    """

    def __init__(
        self,
        wrapping_boundary: float | torch.Tensor = 1.0,
        sigma_min: float = 0.01,
        sigma_max: float = 5.0,
        limit_info_key: str = "num_atoms",
    ):
        super().__init__(
            sigma_min=sigma_min, sigma_max=sigma_max, wrapping_boundary=wrapping_boundary
        )
        self.limit_info_key = limit_info_key

    def std_scaling(self, batch: BatchedData) -> torch.Tensor:
        return batch[self.limit_info_key] ** (-1 / 3)

    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = super().marginal_prob(x, t, batch_idx, batch)
        assert (
            batch is not None
        ), "batch must be provided when using NumAtomsVarianceAdjustedWrappedVESDEMixin"
        std_scale = self.std_scaling(batch)
        std = std * maybe_expand(std_scale, batch_idx, like=std)
        return mean, std

    def prior_sampling(
        self,
        shape: torch.Size | tuple,
        conditioning_data: BatchedData | None = None,
        batch_idx=None,
    ) -> torch.Tensor:
        _super = super()
        assert isinstance(self, DiffSDE) and hasattr(_super, "prior_sampling")
        assert (
            conditioning_data is not None
        ), "batch must be provided when using NumAtomsVarianceAdjustedWrappedVESDEMixin"
        num_atoms = conditioning_data[self.limit_info_key]
        batch_idx = torch.repeat_interleave(
            torch.arange(num_atoms.shape[0], device=num_atoms.device), num_atoms, dim=0
        )
        std_scale = self.std_scaling(conditioning_data)
        # prior sample is randn() * sigma_max, so we need additionally multiply by std_scale to get the correct variance.
        # We call VESDE.prior_sampling (a "grandparent" function) because the super() prior_sampling already does the wrapping,
        # which means we couldn't do the variance adjustment here anymore otherwise.
        prior_sample = DiffVESDE.prior_sampling(self, shape=shape).to(num_atoms.device)
        return self.wrap(prior_sample * maybe_expand(std_scale, batch_idx, like=prior_sample))

    def sde(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = self.marginal_prob(x, t, batch_idx, batch)[1]
        sigma_min = self.marginal_prob(x, torch.zeros_like(t), batch_idx, batch)[1]
        sigma_max = self.marginal_prob(x, torch.ones_like(t), batch_idx, batch)[1]
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(2 * (sigma_max.log() - sigma_min.log()))
        return drift, diffusion
