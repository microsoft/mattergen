# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import cached_property
from pathlib import Path

from huggingface_hub import hf_hub_download

from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer

REFERENCE_REPOSITORY_NAME = "microsoft/mattergen"
REFERENCE_DATASET_DIRECTORY = "data-release/alex-mp"


def _is_gzip_file(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def get_reference_dataset_path(filename: str) -> Path:
    source_checkout_path = (
        Path(__file__).resolve().parents[3] / REFERENCE_DATASET_DIRECTORY / filename
    )
    if _is_gzip_file(source_checkout_path):
        return source_checkout_path
    return Path(
        hf_hub_download(
            repo_id=REFERENCE_REPOSITORY_NAME,
            filename=f"{REFERENCE_DATASET_DIRECTORY}/{filename}",
        )
    )


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
        return LMDBGZSerializer().deserialize(
            get_reference_dataset_path("reference_MP2020correction.gz")
        )

    @cached_property
    def is_ordered(self) -> bool:
        """Returns True if all structures are ordered."""
        return True # Setting it manually to avoid computation at runtime.


class ReferenceTRI2024Correction(ReferenceDataset):
    """Reference dataset using the TRI2024 Energy Correction scheme.
    This dataset contains entries from the Materials Project [https://next-gen.materialsproject.org/]
    and Alexandria [https://next-gen.materialsproject.org/].
    All 845,997 structures are relaxed using the GGA-PBE functional and have energy corrections applied using the TRI2024 scheme.
    """

    def __init__(self):
        super().__init__("TRI2024correction", ReferenceTRI2024Correction.from_preset())

    @classmethod
    def from_preset(cls) -> "ReferenceTRI2024Correction":
        return LMDBGZSerializer().deserialize(
            get_reference_dataset_path("reference_TRI2024correction.gz")
        )

    @cached_property
    def is_ordered(self) -> bool:
        """Returns True if all structures are ordered."""
        return True # Setting it manually to avoid computation at runtime.
