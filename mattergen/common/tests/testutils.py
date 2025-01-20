# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path

import torch
from torch_geometric.data import Batch


def get_mp_20_debug_batch() -> Batch:
    # loads a batch containing the first 64 crystals of the mp_20 training set.
    return torch.load(Path(__file__).resolve().parent / "mp_20_debug_batch.pt")
