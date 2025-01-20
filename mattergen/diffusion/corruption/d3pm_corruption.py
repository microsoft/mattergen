# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Tuple, Union

import torch
from torch_scatter import scatter_add

from mattergen.diffusion.corruption.corruption import B, Corruption, maybe_expand
from mattergen.diffusion.d3pm import d3pm
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.discrete_time import to_discrete_time


class D3PMCorruption(Corruption):
    """D3PM discrete corruption process. Has discret time and discrete (categorical) values."""

    def __init__(
        self,
        d3pm: d3pm.DiscreteDiffusionBase,
        offset: int = 0,
    ):
        super().__init__()
        self.d3pm = d3pm
        # Often, the data is not zero-indexed, so we need to offset the data
        # E.g., if we are dealing with one-based class labels, we might want to offset by 1 to convert from zero-based indices to actual classes.
        self.offset = offset

    @property
    def N(self) -> int:
        """Number of diffusion timesteps i.e. number of noise levels.
        Must match number of noise levels used for sampling. To change this, we'd need to implement continuous-time diffusion for discrete things
        as in e.g. Campbell et al. https://arxiv.org/abs/2205.14987"""
        return self.d3pm.num_steps

    def _to_zero_based(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from non-zero-based indices to zero-based indices."""
        return x - self.offset

    def _to_non_zero_based(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from zero-based indices to non-zero-based indices."""
        return x + self.offset

    @property
    def T(self) -> float:
        """End time of the Corruption process."""
        return 1

    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parameters to determine the marginal distribution of the corruption process, $p_t(x | x_0)$."""
        # plus 1 because t=0 is actually no corruption for D3PM and it has N corruption steps, i.e., values go from 0 to N.
        t_discrete = maybe_expand(to_discrete_time(t, N=self.N, T=self.T), batch_idx) + 1
        _, logits = d3pm.q_sample(
            self._to_zero_based(x.long()), t_discrete, diffusion=self.d3pm, return_logits=True
        )
        return logits, None  # mean: (nodes_per_sample * batch_size, ), std None

    def prior_sampling(
        self,
        shape: Union[torch.Size, Tuple],
        conditioning_data: Optional[BatchedData] = None,
        batch_idx: B = None,
    ) -> torch.Tensor:
        """Generate one sample from the prior distribution, $p_T(x)$."""
        # sample and then add offset to convert to non-zero-based class labels
        return self._to_non_zero_based(self.d3pm.sample_stationary(shape))

    def prior_logp(
        self,
        z: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> torch.Tensor:
        """Compute log-density of the prior distribution.

        Args:
          z: samples, non-zero-based indices, i.e., we first need to subtract the offset
        Returns:
          log probability density
        """
        probs = self.d3pm.stationary_probs(z.shape).to(z.device)
        log_probs = (probs + 1e-8).log()
        log_prob_per_sample = log_probs[:, self._to_zero_based(z.long())]
        log_prob_per_structure = scatter_add(log_prob_per_sample, batch_idx, dim=0)
        return log_prob_per_structure

    def sample_marginal(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: Optional[BatchedData] = None,
    ) -> torch.Tensor:
        """Sample marginal for x(t) given x(0).
        Returns:
          sampled x(t), non-zero-based indices
          where raw_noise is drawn from standard Gaussian
        """
        logits = self.marginal_prob(x=x, t=t, batch_idx=batch_idx, batch=batch)[0]
        sample = torch.distributions.Categorical(logits=logits).sample()
        # samples are zero-based, so we need to add the offset to convert to non-zero-based class labels.
        return self._to_non_zero_based(sample)
