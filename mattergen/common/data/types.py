# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from emmet.core.material import PropertyOrigin

from mattergen.common.utils.globals import PROPERTY_SOURCE_IDS

PropertySourceId = str
TargetProperty = dict[PropertySourceId, int | float | str | Sequence[str]]


@dataclass(frozen=True)
class PropertyValues:
    "A class for storing the values of a property"
    values: np.ndarray
    property_source_doc_id: PropertySourceId
    origins: list[PropertyOrigin] | None = (
        None  # Dictionary for tracking the provenance of properties, emmet-style.
    )

    def __post_init__(self):
        assert self.property_source_doc_id in PROPERTY_SOURCE_IDS, (
            f"property_source_doc_id {self.property_source_doc_id} not found in the database. "
            f"Available property_source_doc_ids: {PROPERTY_SOURCE_IDS}"
        )

    @property
    def n_entries(self) -> int:
        return self.values.shape[0]

    def to_json(self, filename):
        with open(filename, "w") as f:
            json.dump(
                {
                    "values": self.values.tolist(),
                    "property_source_doc_id": self.property_source_doc_id,
                    "origins": self.origins,
                },
                f,
            )

    @classmethod
    def from_json(cls, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        data["values"] = np.array(data["values"])
        return cls(**data)
