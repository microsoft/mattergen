# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from typing import Any, TypeVar

import torch

from mattergen.diffusion.corruption.sde_lib import SDE
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.model_target import ModelTarget

T = TypeVar("T", bound=BatchedData)


def convert_model_out_to_score(
    *,
    model_target: ModelTarget,
    sde: SDE,
    model_out: torch.Tensor,
    batch_idx: torch.LongTensor,
    t: torch.Tensor,
    batch: Any
) -> torch.Tensor:
    """
    Convert a model output to a score, according to the specified model_target.

    model_target: says what the model predicts.
        For example, in RFDiffusion the model predicts clean coordinates;
        in EDM the model predicts the raw noise.
    sde: corruption process
    model_out: model output
    batch_idx: indicates which sample each row of model_out belongs to
    noisy_x: noisy data
    t: diffusion timestep
    batch: noisy batch, ignored except by strange SDEs
    """
    _, std = sde.marginal_prob(
        x=torch.ones_like(model_out),
        t=t,
        batch_idx=batch_idx,
        batch=batch,
    )
    # Note the slack tolerances in test_model_utils.py: the choice of ModelTarget does make a difference.
    if model_target == ModelTarget.score_times_std:
        return model_out / std
    elif model_target == ModelTarget.logits:
        # Not really a score, but logits will be handled downstream.
        return model_out
    else:
        raise NotImplementedError


class NoiseLevelEncoding(torch.nn.Module):
    """
    From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor, shape [batch_size]
        """
        x = torch.zeros((t.shape[0], self.d_model), device=self.div_term.device)
        x[:, 0::2] = torch.sin(t[:, None] * self.div_term[None])
        x[:, 1::2] = torch.cos(t[:, None] * self.div_term[None])
        return self.dropout(x)
