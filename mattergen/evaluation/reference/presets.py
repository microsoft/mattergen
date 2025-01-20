# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from functools import cached_property

from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer


class ReferenceMP2020Correction(ReferenceDataset):
    """Reference dataset using the MP2020 Energy Correction scheme.
    This dataset contains entries from the Materials Project [https://next-gen.materialsproject.org/]
    and Alexandria [https://next-gen.materialsproject.org/].
    All 845,997 structures are relaxed using the GGA-PBE functional and have energy corrections applied using the MP2020 scheme.
    """

    def __init__(self):
        super().__init__("MP2020correction", ReferenceMP2020Correction.from_preset())

    @classmethod
    def from_preset(cls) -> "ReferenceMP2020Correction":
        current_dir = Path(__file__).parent
        return LMDBGZSerializer().deserialize(
            f"{current_dir}/../../../data-release/alex-mp/reference_MP2020correction.gz"
        )

    @cached_property
    def is_ordered(self) -> bool:
        """Returns True if all structures are ordered."""
        return True # Setting it manually to avoid computation at runtime.
