# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
from copy import deepcopy
from functools import cached_property
from typing import Literal, Type

import numpy as np
import numpy.typing
from pandas import DataFrame

from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils.metrics_structure_summary import MetricsStructureSummary


class BaseMetricsCapability:
    """Base class for capabilities."""

    name: str = "base_capability"

    def __init__(
        self, structure_summaries: list[MetricsStructureSummary], n_failed_jobs: int = 0
    ) -> None:
        assert len(structure_summaries) > 0, "No data provided."
        self._structure_summaries = structure_summaries
        self.n_failed_jobs = n_failed_jobs

    @property
    def total_submitted_jobs(self) -> int:
        return len(self.dataset) + self.n_failed_jobs

    @cached_property
    def dataset(self) -> ReferenceDataset:
        """
        Returns a ReferenceDataset. While not all capabilities require energies,
        the entry IDs are useful to keep track of entry IDs.
        """
        data_entries = [deepcopy(s.entry) for s in self._structure_summaries]
        for i, e in enumerate(data_entries):
            e.entry_id = i

        return ReferenceDataset.from_entries("data_entries", data_entries)

    @abc.abstractmethod
    def as_dataframe(self) -> DataFrame:
        """Returns a pandas DataFrame containing information about this capability."""


class BaseMetric:
    """Abstract base class for metrics."""

    required_capabilities: tuple[Type[BaseMetricsCapability], ...]

    @property
    def name(self) -> str:
        return "base_metric"

    @property
    def description(self) -> str:
        raise NotImplementedError

    @cached_property
    def value(self) -> float | int:
        raise NotImplementedError


class BaseAggregateMetric(BaseMetric):
    """Abstract base class for aggregate metrics."""

    aggregation_method: Literal[
        "mean", "nanmean",
    ] = "not implemented"

    @property
    def pre_aggregation_name(self) -> str:
        return "base_metric"

    @abc.abstractmethod
    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        """Compute metric values for each sample in the dataset."""

    @cached_property
    def pre_aggregation_values(self) -> numpy.typing.NDArray:
        """Metric values for each sample in the dataset before aggregation."""
        return self.compute_pre_aggregation_values()

    @cached_property
    def value(self) -> float | int:
        values = self.pre_aggregation_values
        if self.aggregation_method == "mean":
            return values.mean()
        elif self.aggregation_method == "nanmean":
            return np.nanmean(values)
        else:
            raise ValueError(f"Unknown aggregation method {self.aggregation_method}")
