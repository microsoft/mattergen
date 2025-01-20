# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List

import pytest

from mattergen.common.utils.globals import MAX_ATOMIC_NUM
from mattergen.property_embeddings import ChemicalSystemMultiHotEmbedding


@pytest.mark.parametrize("chemical_system", [["Li", "O"], ["Li", "O", "F"], ["C", "O", "H"]])
def test_chemical_system_to_multi_hot(chemical_system: List[str]) -> None:
    # test that multi-hot encoding executes without error and is correct shape and has correct number of 1s in columns

    multi_hot_encoding = ChemicalSystemMultiHotEmbedding._sequence_to_multi_hot(
        x=chemical_system, device="cpu"
    )

    assert multi_hot_encoding.shape == (
        1,
        MAX_ATOMIC_NUM + 1,
    )
    assert multi_hot_encoding.sum() == len(chemical_system)
