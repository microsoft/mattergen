# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, cast

import torch

from mattergen.diffusion.corruption.corruption import Corruption
from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.corruption.sde_lib import ScoreFunction
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.discrete_time import to_discrete_time
from mattergen.diffusion.sampling.predictors import Predictor
from mattergen.diffusion.sampling.predictors_correctors import SampleAndMean


class D3PMAncestralSamplingPredictor(Predictor):
    """
    Ancestral sampling predictor for D3PM.
    """

    def __init__(
        self,
        *,
        corruption: D3PMCorruption,
        score_fn: ScoreFunction,
        predict_x0: bool = True,
    ):
        super().__init__(corruption=corruption, score_fn=score_fn)
        # if True, self.denoiser returns p(x_0|x_t), otherwise p(x_{t-1}|x_t)
        self.predict_x0 = predict_x0

    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        return isinstance(corruption, D3PMCorruption)

    @property
    def N(self) -> int:
        self.corruption = cast(D3PMCorruption, self.corruption)  # mypy
        return self.corruption.N

    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
        batch: Optional[BatchedData],
    ) -> SampleAndMean:
        """
        Takes the atom coordinates, cell vectors and atom types at time t and
        returns the atom types at time t-1, sampled using the learned reverse
        atom diffusion model.

        Look at https://github.com/google-research/google-research/blob/master/d3pm/text/diffusion.py

        lines 3201-3229. NOTE: we do implement the taking the softmax of the initial
        sample as per 3226-3227. This could be to avoid weird behaving for picking
        initial states that happened to have very low probability in latent space.
        Try adding if there proves to be a problem generating samples.
        """
        # t is  continuous, needs to be integer
        t = to_discrete_time(t=t, N=self.N, T=self.corruption.T)

        class_logits = score

        assert isinstance(self.corruption, D3PMCorruption)

        # sample from categorical distribution
        x_sample = self.corruption._to_non_zero_based(
            torch.distributions.Categorical(logits=class_logits).sample()
        )

        # convert logit output to normalized probabilities
        class_probs = torch.softmax(class_logits, dim=-1)

        # get expected atom type from categorical distribution
        class_expected = self.corruption._to_non_zero_based(torch.argmax(class_probs, dim=-1))

        if self.predict_x0:
            # if self.predict_x0, the model predicts p(x_0|x_t), not p(x_{t-1}|x_t)
            # We need to evaluate p(x_{t-1}|x_t) by Eq 4. in https://arxiv.org/pdf/2107.03006v1.pdf
            assert isinstance(self.corruption, D3PMCorruption)
            class_logits, _ = self.corruption.d3pm.sample_and_compute_posterior_q(
                x_0=class_probs,
                t=t[batch_idx].to(torch.long),  # requires torch.long or torch.int32
                make_one_hot=False,
                samples=self.corruption._to_zero_based(
                    x
                ),  # d3pm expects 0 offset atom type integers
                return_logits=True,
            )

            x_sample = self.corruption._to_non_zero_based(
                torch.distributions.Categorical(logits=class_logits).sample()
            )

            # get expected atom type
            class_expected = self.corruption._to_non_zero_based(
                torch.argmax(torch.softmax(class_logits.to(class_probs.dtype), dim=-1), dim=-1)
            )

        # (sampled states), (expected states)
        return x_sample, class_expected
