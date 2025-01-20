# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum
from typing import Mapping, Union


class ModelTarget(Enum):
    """Specifies what the score model is trained to predict.
    Only relevant for fields that are corrupted with an SDE."""

    score_times_std = "score_times_std"  # Predict -z where z is gaussian noise with unit variance used to corrupt the data
    logits = "logits"  # Predict logits for a categorical variable


ModelTargets = Mapping[str, Union[ModelTarget, str]]
