# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Adapted from https://github.com/yang-song/score_sde_pytorch which is released under Apache license.

Key changes:
- Introduced batch_idx argument to work with graph-like data (e.g. molecules)
- Introduced `..._given_score` methods so that multiple fields can be sampled at once using a shared score model. See PredictorCorrector for how this is used.
"""

import abc
import logging

import torch

from mattergen.diffusion.corruption.corruption import Corruption
from mattergen.diffusion.corruption.sde_lib import SDE, ScoreFunction, check_score_fn_defined
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.sampling.predictors_correctors import SampleAndMean, Sampler
from mattergen.diffusion.wrapped.wrapped_sde import WrappedSDEMixin

logger = logging.getLogger(__name__)


class Predictor(Sampler):
    """The abstract class for something that takes x_t and predicts x_{t-dt},
    where t is diffusion timestep."""

    def __init__(
        self,
        corruption: Corruption,
        score_fn: ScoreFunction | None,
    ):
        super().__init__(corruption, score_fn=score_fn)

    def update_fn(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        batch: BatchedData | None,
    ) -> SampleAndMean:
        """One update of the predictor.

        Args:
          x: current state
          t: timesteps
          batch_idx: indicates which sample each row of x belongs to

        Returns:
           (sampled next state, mean next state)
        """
        check_score_fn_defined(self.score_fn, "update_given_score")
        assert self.score_fn is not None
        score = self.score_fn(x=x, t=t, batch_idx=batch_idx)
        return self.update_given_score(
            x=x, t=t, dt=dt, batch_idx=batch_idx, score=score, batch=batch
        )

    @abc.abstractmethod
    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
        batch: BatchedData | None,
    ) -> SampleAndMean:
        pass


class AncestralSamplingPredictor(Predictor):
    """Suitable for all linear SDEs.

    This predictor is derived by converting the score prediction to a prediction of x_0 given x_t, and then
    sampling from the conditional distribution of x_{t-dt} given x_0 and x_t according to the corruption process.
    It corresponds to equation (47) in Song et al. for VESDE (https://openreview.net/forum?id=PxTIG12RRHS)
    and equation (7) in Ho et al. for VPSDE (https://arxiv.org/abs/2006.11239)

    In more detail: suppose the SDE has marginals x_t ~ N(alpha_t *x_0, sigma_t**2)

    We estimate x_0 as follows:
    x_0 \approx (x_t + sigma_t^2 * score) / alpha_t

    For any s < t, the forward corruption process implies that
    x_t| x_s ~ N(alpha_t/alpha_s * x_s, sigma_t^2 - sigma_s^2 * alpha_t^2 / alpha_s^2)

    Now go away and do some algebra to get the mean and variance of x_s given x_t
    and x_0, and you will get the coefficients in the `update_given_score` method below.

    """

    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
        batch: BatchedData | None,
    ) -> SampleAndMean:
        x_coeff, score_coeff, std = self._get_coeffs(
            x=x,
            t=t,
            dt=dt,
            batch_idx=batch_idx,
            batch=batch,
        )
        # Sample random noise.
        z = torch.randn_like(x_coeff)

        mean = x_coeff * x + score_coeff * score
        sample = mean + std * z

        return sample, mean

    def _get_coeffs(self, x, t, dt, batch_idx, batch):
        """
        Compute coefficients for ancestral sampling.
        This is in a separate method to make it easier to test."""
        sde = self.corruption
        assert isinstance(sde, SDE)

        # Previous timestep
        s = t + dt

        alpha_t, sigma_t = sde.mean_coeff_and_std(x=x, t=t, batch_idx=batch_idx, batch=batch)
        if batch_idx is None:
            is_time_zero = s <= 0
        else:
            is_time_zero = s[batch_idx] <= 0
        alpha_s, sigma_s = sde.mean_coeff_and_std(x=x, t=s, batch_idx=batch_idx, batch=batch)
        sigma_s[is_time_zero] = 0

        # If you are trying to match this up with algebra in papers, it may help to
        # notice that for VPSDE, sigma2_t_given_s == 1 - alpha_t_given_s**2, except
        # that alpha_t_given_s**2 is clipped.
        sigma2_t_given_s = sigma_t**2 - sigma_s**2 * alpha_t**2 / alpha_s**2
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
        std = sigma_t_given_s * sigma_s / sigma_t

        # Clip alpha_t_given_s so that we do not divide by zero.
        min_alpha_t_given_s = 0.001
        alpha_t_given_s = alpha_t / alpha_s
        if torch.any(alpha_t_given_s < min_alpha_t_given_s):
            # If this warning is raised, you probably should change something: either modify your noise schedule
            # so that the diffusion coefficient does not blow up near sde.T, or only denoise from sde.T - eps,
            # rather than sde.T.
            logger.warning(
                f"Clipping alpha_t_given_s to {min_alpha_t_given_s} to avoid divide-by-zero. You should probably change something else to avoid this."
            )
            alpha_t_given_s = torch.clip(alpha_t_given_s, min_alpha_t_given_s, 1)

        score_coeff = sigma2_t_given_s / alpha_t_given_s

        x_coeff = 1.0 / alpha_t_given_s

        std[is_time_zero] = 0

        return x_coeff, score_coeff, std

    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        return super().is_compatible(corruption) and not isinstance(corruption, WrappedSDEMixin)
