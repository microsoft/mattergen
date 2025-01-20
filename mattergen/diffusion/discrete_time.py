# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


def to_discrete_time(t: torch.Tensor, N: int, T: float) -> torch.LongTensor:
    """Convert continuous time to integer timestep.

    Args:
        t: continuous time between 0 and T
        N: number of timesteps
        T: max time
    Returns:
        Integer timesteps between 0 and N-1
    """
    return ((t * (N - 1)) / T).long()
