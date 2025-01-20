# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from typing import Generic, TypeVar

import torch

from mattergen.diffusion.data.batched_data import BatchedData

Diffusable = TypeVar("Diffusable", bound=BatchedData)


class ScoreModel(torch.nn.Module, Generic[Diffusable], abc.ABC):
    """Abstract base class for score models."""

    @abc.abstractmethod
    def forward(self, x: Diffusable, t: torch.Tensor) -> Diffusable:
        """Args:
        x: batch of noisy data
        t: timestep. Shape (batch_size, 1)
        """
        ...
