"""
Copyright 2020 The Google Research Authors.
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Based on code from https://github.com/yang-song/score_sde_pytorch
which is released under Apache licence.

Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Key changes:
- Adapted to work on batched pytorch_geometric style data
- Added '...given_score' methods so that score for a composite
state can be calculated in single forward pass of a shared score model,
and the scores for different fields then forwarded to the different reverse SDEs.
"""

import abc
from typing import Callable, Optional, Protocol, Tuple, Union

import numpy as np
import torch
from torch_scatter import scatter_add

from mattergen.diffusion.corruption.corruption import B, Corruption, maybe_expand
from mattergen.diffusion.data.batched_data import BatchedData


class ScoreFunction(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
    ) -> torch.Tensor:
        """Calculate score.

        Args:
            x: Samples at which the score should be calculated. Shape [num_nodes, ...]
            t: Timestep for each sample. Shape [num_samples,]
            batch_idx: Indicates which sample each row of x belongs to. Shape [num_nodes,]

        """
        pass


class SDE(Corruption):
    """Corruption using a stochastic differential equation."""

    @abc.abstractmethod
    def sde(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns drift f and diffusion coefficient g such that dx = f * dt + g * sqrt(dt) * standard Gaussian"""
        pass  # drift: (nodes_per_sample * batch_size, num_features), diffusion (batch_size,)

    @abc.abstractmethod
    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and standard deviation of the marginal distribution of the SDE, $p_t(x)$."""
        pass  # mean: (nodes_per_sample * batch_size, num_features), std: (nodes_per_sample * batch_size, 1)

    def mean_coeff_and_std(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mean coefficient and standard deviation of marginal distribution at time t."""
        return self.marginal_prob(
            torch.ones_like(x), t, batch_idx, batch
        )  # mean_coeff: same shape as x, std: same shape as x

    def sample_marginal(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> torch.Tensor:
        """Sample marginal for x(t) given x(0).
        Returns:
          sampled x(t)
        """
        mean, std = self.marginal_prob(x=x, t=t, batch_idx=batch_idx, batch=batch)
        z = torch.randn_like(x)

        return mean + std * z


class BaseVPSDE(SDE):
    """Base class for variance-preserving SDEs of the form
            dx = - 0.5 * beta_t * x * dt + sqrt(beta_t) * z * sqrt(dt)
    where z is unit Gaussian noise, or equivalently
            dx = - 0.5 * beta_t *x * dt + sqrt(beta_t) * dW

    """

    @abc.abstractmethod
    def beta(self, t: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def _marginal_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """This should be implemented to compute exp(-0.5 * int_0^t beta(s) ds). See equation (29) of Song et al."""
        ...

    @property
    def T(self) -> float:
        return 1.0

    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_coeff = self._marginal_mean_coeff(t)
        mean = maybe_expand(mean_coeff, batch_idx, x) * x
        std = maybe_expand(torch.sqrt(1.0 - mean_coeff**2), batch_idx, x)
        return mean, std

    def prior_sampling(
        self,
        shape: Union[torch.Size, Tuple],
        conditioning_data: Optional[BatchedData] = None,
        batch_idx: B = None,
    ) -> torch.Tensor:
        return torch.randn(*shape)

    def prior_logp(
        self,
        z: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> torch.Tensor:
        return unit_gaussian_logp(z, batch_idx)

    def sde(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        beta_t = self.beta(t)
        drift = -0.5 * maybe_expand(beta_t, batch_idx, x) * x
        diffusion = maybe_expand(torch.sqrt(beta_t), batch_idx, x)
        return drift, diffusion


class VPSDE(BaseVPSDE):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20):
        """Variance-preserving SDE with drift coefficient changing linearly over time."""
        super().__init__()
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def _marginal_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        return torch.exp(log_mean_coeff)


def unit_gaussian_logp(z: torch.Tensor, batch_idx: B = None) -> torch.Tensor:
    shape = z.shape
    N = np.prod(shape[1:])
    if batch_idx is None:
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=tuple(range(1, z.ndim))) / 2.0
    else:
        if z.ndim > 2:
            raise NotImplementedError

        logps = -N / 2.0 * np.log(2 * np.pi) - scatter_add(torch.sum(z**2, dim=1), batch_idx) / 2.0

    return logps


class VESDE(SDE):
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0):
        """Construct a Variance Exploding SDE.

        The marginal standard deviation grows exponentially from sigma_min to sigma_max.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
        """
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @property
    def T(self) -> float:
        return 1.0

    def sde(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = maybe_expand(
            sigma
            * torch.sqrt(
                torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device)
            ),
            batch_idx,
            x,
        )
        return drift, diffusion

    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        std = maybe_expand(self.sigma_min * (self.sigma_max / self.sigma_min) ** t, batch_idx, x)
        mean = x
        return mean, std

    def prior_sampling(
        self,
        shape: Union[torch.Size, Tuple],
        conditioning_data: Optional[BatchedData] = None,
        batch_idx: B = None,
    ) -> torch.Tensor:
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(
        self,
        z: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> torch.Tensor:
        shape = z.shape
        N = np.prod(shape[1:])
        if batch_idx is not None:
            return -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - scatter_add(
                torch.sum(z**2, dim=1), batch_idx
            ) / (2 * self.sigma_max**2)
        else:
            return -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(
                z**2, dim=tuple(range(1, z.ndim))
            ) / (2 * self.sigma_max**2)


def check_score_fn_defined(score_fn: Optional[Callable], fn_name_given_score: str):
    """Check that a reverse SDE has a score_fn. Give a useful error message if not."""
    if score_fn is None:
        raise ValueError(
            f"This reverse SDE does not know its score_fn. You must either a) pass a score_fn when you construct this reverse SDE or b) call {fn_name_given_score} instead."
        )
