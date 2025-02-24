# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from mattergen.common.diffusion import corruption as sde_lib
from mattergen.common.utils.data_utils import compute_lattice_polar_decomposition
from mattergen.diffusion.corruption.corruption import Corruption, maybe_expand
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.sampling import predictors_correctors as pc
from mattergen.diffusion.sampling.predictors import AncestralSamplingPredictor

SampleAndMean = tuple[torch.Tensor, torch.Tensor]


class LatticeAncestralSamplingPredictor(AncestralSamplingPredictor):
    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        _super = super()
        assert hasattr(_super, "is_compatible")
        return _super.is_compatible(corruption) or isinstance(corruption, sde_lib.LatticeVPSDE)

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
        # mean = (x + score * beta**2 - limit_mean)/(1-beta) + limit_mean
        # <=> mean = x / (1-beta) + score * beta**2 / (1-beta) + limit_mean * (1 - 1/(1-beta))
        # => mean_coeff = 1 - x_coeff = 1 - 1/(1-beta)
        mean_coeff = 1 - x_coeff
        # Sample random noise.
        z = sde_lib.make_noise_symmetric_preserve_variance(torch.randn_like(x_coeff))
        assert hasattr(self.corruption, "get_limit_mean")  # mypy
        mean = (
            x_coeff * x
            + score_coeff * score
            + mean_coeff * self.corruption.get_limit_mean(x=x, batch=batch)
        )
        sample = mean + std * z
        return sample, mean


# create a langevin corrector that accepts LatticeVPSDE
class LatticeLangevinDiffCorrector(pc.LangevinCorrector):
    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        _super = super()
        assert hasattr(_super, "is_compatible")
        return _super.is_compatible(corruption) or isinstance(corruption, sde_lib.LatticeVPSDE)

    def step_given_score(
        self,
        *,
        x: torch.Tensor,
        batch_idx: torch.LongTensor | None,
        score: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
    ) -> SampleAndMean:
        assert isinstance(self.corruption, sde_lib.LatticeVPSDE)
        alpha = self.get_alpha(t, dt=dt)
        snr = self.snr
        noise = torch.randn_like(x)
        noise = sde_lib.make_noise_symmetric_preserve_variance(noise)

        # [batch_size, ] or [num_atoms, ] if batch_idx is not None
        grad_norm_square = torch.square(score).reshape(score.shape[0], -1).sum(dim=1)
        noise_norm_square = torch.square(noise).reshape(noise.shape[0], -1).sum(dim=1)
        # Average over items, leading to scalars.
        grad_norm = grad_norm_square.sqrt().mean()
        noise_norm = noise_norm_square.sqrt().mean()

        # If gradient is zero (i.e., we are sampling from an improper distribution that's flat over the whole of R^n)
        # the step_size blows up. Clip step_size to avoid this.
        # The EGNN reports zero scores when there are no edges between nodes.
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        step_size = torch.minimum(step_size, self.max_step_size)
        step_size[grad_norm == 0, :] = self.max_step_size
        step_size = maybe_expand(step_size, batch_idx, score)
        mean = x + step_size * score
        x = mean + torch.sqrt(step_size * 2) * noise

        x = compute_lattice_polar_decomposition(x)
        mean = compute_lattice_polar_decomposition(mean)
        return x, mean
