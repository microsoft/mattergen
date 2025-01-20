# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from numpy.typing import NDArray
from mattergen.common.data.dataset import BaseDataset

# Dataset transforms
# These transforms are used to modify the dataset in various ways, such as filtering out
# structures with missing properties.

def is_nan(value: NDArray) -> NDArray:
    if value.dtype.kind == "U":
        # str dtype, e.g., space group
        return np.zeros(value.shape, dtype=bool)
    return np.isnan(value)

def filter_sparse_properties(dataset: BaseDataset) -> BaseDataset:
    """
    Filter out structures with missing properties.
    Returns a new dataset with only structures that have all properties.
    """
    if len(dataset.properties) == 0:
        return dataset
    indices_with_all_properties = np.where(
        np.all([~is_nan(val) for val in dataset.properties.values()], axis=0)
    )[0]
    return dataset.subset(indices=indices_with_all_properties)


def repeat(dataset: BaseDataset, n: int) -> BaseDataset:
    """
    Repeat the dataset n times.
    """
    return dataset.repeat(n)
