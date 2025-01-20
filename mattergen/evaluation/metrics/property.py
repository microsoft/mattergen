# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing
from pandas import DataFrame

from mattergen.evaluation.metrics.core import BaseAggregateMetric, BaseMetric, BaseMetricsCapability
from mattergen.evaluation.metrics.energy import EnergyMetricsCapability
from mattergen.evaluation.metrics.structure import StructureMetricsCapability
from mattergen.evaluation.utils.metrics_structure_summary import MetricsStructureSummary
from mattergen.evaluation.utils.utils import PropertyConstraint


class PropertyMetricsCapability(BaseMetricsCapability):
    name: str = "property_capability"

    """Capability for computing property metrics."""

    def __init__(
        self,
        structure_summaries: list[MetricsStructureSummary],
        property_constraints: dict[str, PropertyConstraint] | None = None,
        n_failed_jobs: int = 0,
    ) -> None:
        super().__init__(structure_summaries=structure_summaries, n_failed_jobs=n_failed_jobs)
        self.property_constraints = property_constraints

    @cached_property
    def properties(self) -> dict[str, numpy.typing.NDArray]:
        props = list(self._structure_summaries[0].properties)
        assert all(
            set(s.properties.keys()) == set(props) for s in self._structure_summaries
        ), "Inconsistent property data."
        return {
            prop: np.array([s.properties[prop] for s in self._structure_summaries])
            for prop in props
        }

    @property
    def satisfies_property_constraints(self) -> numpy.typing.NDArray[np.bool_]:
        """
        Returns a boolean mask of the same length as structure_summaries
        indicating whether each entry satisfies the property constraints.
        """

        def _satisfies_property_constraint(
            values: np.array, constraint: PropertyConstraint
        ) -> numpy.typing.NDArray[np.bool_]:
            # Check if values are within (min, max) constraints
            mask = True if constraint[0] is None else values >= constraint[0]
            mask &= True if constraint[1] is None else values <= constraint[1]
            return mask

        assert self.property_constraints, "No property constraints specified."

        assert all(
            key in self.properties for key in self.property_constraints
        ), f"Property data and constraints do not match: {list(self.properties)} vs. {list(self.property_constraints)}."

        return np.all(
            np.array(
                [
                    _satisfies_property_constraint(self.properties[key], constraint)
                    for key, constraint in self.property_constraints.items()
                ],
                dtype=bool,
            ),
            axis=0,
        )

    def as_dataframe(self) -> DataFrame:
        data = {str(k): v for k, v in self.properties.items()}
        if self.property_constraints:
            data.update({"satisfies_property_constraints": self.satisfies_property_constraints})

        return DataFrame(
            data=data,
            index=[e.entry_id for e in self.dataset],
        )


# -----------------------------#
# Metrics
# -----------------------------#


@dataclass(frozen=True)
class BasePropertyMetric(BaseMetric):
    # Use for metrics that have access to structure, energy and property data.
    # In principle, we could define metrics classes with fewer required capabilities, but this is not necessary for now.
    required_capabilities = (
        StructureMetricsCapability,
        EnergyMetricsCapability,
        PropertyMetricsCapability,
    )

    @property
    def name(self) -> str:
        return "base_property_metric"

    def __init__(
        self,
        structure_capability: StructureMetricsCapability,
        energy_capability: EnergyMetricsCapability,
        property_capability: PropertyMetricsCapability,
        **kwargs,  # eat up unused kwargs (i.e., other capabilities)
    ):
        self.structure_capability = structure_capability
        self.energy_capability = energy_capability
        self.property_capability = property_capability
        self.reference_dataset = self.energy_capability.reference_dataset


class FracStableStructuresWithProperties(BasePropertyMetric, BaseAggregateMetric):
    name = "frac_stable_structures_with_properties"
    pre_aggregation_name = "stable_with_properties"

    @property
    def description(self) -> str:
        return (
            f"Fraction of stable structures in sampled data within {self.energy_capability.stability_threshold} (eV/atom) "
            + f"above convex hull of {self.reference_dataset.name} and that satisfy target property constraints."
        )

    @cached_property
    def value(self) -> float:
        return self.pre_aggregation_values.sum() / self.energy_capability.total_submitted_jobs

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        return (
            self.energy_capability.is_stable
            & self.property_capability.satisfies_property_constraints
        )


class FracNovelUniqueStableStructuresWithProperties(BasePropertyMetric, BaseAggregateMetric):
    name = "frac_novel_unique_stable_structures_with_properties"
    pre_aggregation_name = "novel_unique_stable_with_properties"

    @property
    def description(self) -> str:
        return (
            f"Fraction of novel unique stable structures in sampled data within {self.energy_capability.stability_threshold} (eV/atom) "
            + f"above convex hull of {self.reference_dataset.name} and that satisfy target property constraints."
        )

    @cached_property
    def value(self) -> float:
        return self.pre_aggregation_values.sum() / self.property_capability.total_submitted_jobs

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        return (
            self.structure_capability.is_novel
            & self.structure_capability.is_unique
            & self.energy_capability.is_stable
            & self.property_capability.satisfies_property_constraints
        )
