# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Tuple, Union

import torch

from mattergen.diffusion.corruption.sde_lib import SDE, VESDE, VPSDE
from mattergen.diffusion.data.batched_data import BatchedData

B = Optional[torch.LongTensor]


def wrap_at_boundary(x: torch.Tensor, wrapping_boundary: float) -> torch.Tensor:
    """Wrap x at the boundary given by wrapping_boundary.
    Args:
      x: tensor of shape (batch_size, dim)
      wrapping_boundary: float): wrap at [0, wrapping_boundary] in all dimensions.
    Returns:
      wrapped_x: tensor of shape (batch_size, dim)
    """
    return torch.remainder(
        x, wrapping_boundary
    )  # remainder is the same as mod, but works with negative numbers.


class WrappedSDEMixin:
    def sample_marginal(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: torch.LongTensor = None,
        batch: Optional[BatchedData] = None,
    ) -> torch.Tensor:
        _super = super()
        assert (
            isinstance(self, SDE)
            and hasattr(_super, "sample_marginal")
            and hasattr(self, "wrapping_boundary")
        )
        if (x > self.wrapping_boundary).any() or (x < 0).any():
            # Values outside the wrapping boundary are valid in principle, but could point to an issue in the data preprocessing,
            # as typically we assume that the input data is inside the wrapping boundary (e.g., angles between 0 and 2*pi).
            print("Warning: Wrapped SDE has received input outside of the wrapping boundary.")
        noisy_x = _super.sample_marginal(x=x, t=t, batch_idx=batch_idx, batch=batch)
        return self.wrap(noisy_x)

    def prior_sampling(
        self,
        shape: Union[torch.Size, Tuple],
        conditioning_data: Optional[BatchedData] = None,
        batch_idx: B = None,
    ) -> torch.Tensor:
        _super = super()
        assert isinstance(self, SDE) and hasattr(_super, "prior_sampling")
        return self.wrap(_super.prior_sampling(shape=shape, conditioning_data=conditioning_data))

    def wrap(self, x):
        assert isinstance(self, SDE) and hasattr(self, "wrapping_boundary")
        return wrap_at_boundary(x, self.wrapping_boundary)


class WrappedVESDE(WrappedSDEMixin, VESDE):
    def __init__(
        self,
        wrapping_boundary: float = 1.0,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
    ):
        super().__init__(sigma_min=sigma_min, sigma_max=sigma_max)
        self.wrapping_boundary = wrapping_boundary


class WrappedVPSDE(WrappedSDEMixin, VPSDE):
    def __init__(
        self,
        wrapping_boundary: float = 1.0,
        beta_min: float = 0.1,
        beta_max: float = 20,
    ):
        super().__init__(beta_min=beta_min, beta_max=beta_max)
        self.wrapping_boundary = wrapping_boundary
