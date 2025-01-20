# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from collections.abc import Iterable, Sequence
from functools import cached_property
from inspect import getmembers, isclass
from pathlib import Path
from typing import Literal, Sequence, Type, TypeVar

import numpy.typing
import pandas as pd
from monty.serialization import dumpfn
from pandas import DataFrame
from pymatgen.core.structure import Structure
from pymatgen.entries.compatibility import Compatibility, MaterialsProject2020Compatibility
from typing_extensions import Self

import mattergen.evaluation.metrics.energy as energy_metrics
import mattergen.evaluation.metrics.property as property_metrics
import mattergen.evaluation.metrics.structure as structure_metrics
from mattergen.evaluation.metrics.core import BaseAggregateMetric, BaseMetric, BaseMetricsCapability
from mattergen.evaluation.metrics.energy import EnergyMetricsCapability, MissingTerminalsError
from mattergen.evaluation.metrics.property import PropertyMetricsCapability
from mattergen.evaluation.metrics.structure import StructureMetricsCapability
from mattergen.evaluation.reference.presets import ReferenceMP2020Correction
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils.globals import DEFAULT_STABILITY_THRESHOLD
from mattergen.evaluation.utils.logging import logger
from mattergen.evaluation.utils.metrics_structure_summary import (
    MetricsStructureSummary,
    get_metrics_structure_summaries,
)
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DisorderedStructureMatcher,
    OrderedStructureMatcher,
)
from mattergen.evaluation.utils.utils import PropertyConstraint

T = TypeVar("T")


def unique_item(iterable: Iterable[T]) -> T:
    """returns the content of a sequence containing a single item."""
    lst = list(iterable)
    assert len(lst) == 1, f"Tried to call unique_item, but {lst} contains {len(lst)} items."
    return lst[0]


class MetricsEvaluator:
    """
    This class is used to evaluate a set of metrics on a set of structures.
    """

    def __init__(self, capabilities: Sequence[BaseMetricsCapability]):
        assert len(capabilities) > 0, "At least one capability is required."
        self.capabilities = capabilities

        self._metrics: dict[
            Type[BaseMetric], BaseMetric
        ] = {}  # use to cache previously instantiated metrics

    @classmethod
    def from_structures(
        cls,
        structures: list[Structure],
        reference: ReferenceDataset | None = None,
        structure_matcher: OrderedStructureMatcher
        | DisorderedStructureMatcher = DefaultDisorderedStructureMatcher(),
        n_failed_jobs: int = 0,
    ) -> Self:
        """Instantiate MetricsEvaluator from a list of structures. This is useful for computing structure-based metrics."""

        if reference is None:
            print("No reference dataset provided. Using MP2020 correction dataset as reference.")
            reference = ReferenceMP2020Correction()

        structure_summaries = [MetricsStructureSummary.from_structure(s) for s in structures]
        structure_capability = StructureMetricsCapability(
            structure_summaries=structure_summaries,
            reference_dataset=reference,
            structure_matcher=structure_matcher,
            n_failed_jobs=n_failed_jobs,
        )
        return cls(capabilities=[structure_capability])

    @classmethod
    def from_structures_and_energies(
        cls,
        structures: list[Structure],
        energies: list[float],
        reference: ReferenceDataset | None = None,
        properties: dict[str, list[float]] | None = None,
        property_constraints: dict[str, PropertyConstraint] | None = None,
        original_structures: list[Structure] | None = None,
        stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
        structure_matcher: OrderedStructureMatcher
        | DisorderedStructureMatcher = DefaultDisorderedStructureMatcher(),
        energy_correction_scheme: Compatibility = MaterialsProject2020Compatibility(),
        n_failed_jobs: int = 0,
    ) -> Self:

        if reference is None:
            print("No reference dataset provided. Using MP2020 correction as reference.")
            reference = ReferenceMP2020Correction()

        structure_summaries = get_metrics_structure_summaries(
            structures=structures,
            energies=energies,
            properties=properties,
            original_structures=original_structures,
            energy_correction_scheme=energy_correction_scheme,
        )

        return cls.from_structure_summaries(
            structure_summaries=structure_summaries,
            reference=reference,
            stability_threshold=stability_threshold,
            property_constraints=property_constraints,
            structure_matcher=structure_matcher,
            n_failed_jobs=n_failed_jobs,
        )

    @classmethod
    def from_structure_summaries(
        cls,
        structure_summaries: list[MetricsStructureSummary],
        reference: ReferenceDataset | None = None,
        stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
        property_constraints: dict[str, PropertyConstraint] | None = None,
        structure_matcher: OrderedStructureMatcher
        | DisorderedStructureMatcher = DefaultDisorderedStructureMatcher(),
        n_failed_jobs: int = 0,
    ) -> Self:

        if reference is None:
            print("No reference dataset provided. Using MP2020 correction as reference.")
            reference = ReferenceMP2020Correction()

        capabilities: list[BaseMetricsCapability] = []

        if reference is not None:
            structure_capability = StructureMetricsCapability(
                structure_summaries=structure_summaries,
                reference_dataset=reference,
                structure_matcher=structure_matcher,
                n_failed_jobs=n_failed_jobs,
            )
            capabilities.append(structure_capability)
            try:
                energy_capability = EnergyMetricsCapability(
                    structure_summaries=structure_summaries,
                    reference_dataset=reference,
                    stability_threshold=stability_threshold,
                    n_failed_jobs=n_failed_jobs,
                )
                capabilities.append(energy_capability)
            except MissingTerminalsError:
                # if there are missing terminal systems in the reference dataset, we simply don't
                # add the energy capability as we can still compute structure metrics.
                pass

        if any([c for c in structure_summaries if c.properties]):
            property_capability = PropertyMetricsCapability(
                structure_summaries=structure_summaries,
                property_constraints=property_constraints,
                n_failed_jobs=n_failed_jobs,
            )
            capabilities.append(property_capability)

        return cls(capabilities=capabilities)

    @cached_property
    def available_capability_types(self) -> frozenset[Type[BaseMetricsCapability]]:
        return frozenset([type(cap) for cap in self.capabilities])

    @cached_property
    def available_metrics(self) -> list[Type[BaseMetric]]:
        return [
            metric
            for metric in get_all_metrics_classes()
            if all(cap in self.available_capability_types for cap in metric.required_capabilities)
        ]

    @property
    def is_unique(self) -> numpy.typing.NDArray:
        return self.structure_capability.is_unique

    @property
    def is_novel(self) -> numpy.typing.NDArray:
        return self.structure_capability.is_novel

    @property
    def matches_in_reference(self) -> dict[int, list[str]]:
        return self.structure_capability.matches_in_reference

    @property
    def is_in_reference(self) -> tuple[numpy.typing.NDArray]:
        return self.structure_capability.is_in_reference

    @property
    def is_stable(self) -> numpy.typing.NDArray:
        return self.energy_capability.is_stable

    @property
    def is_self_consistent_stable(self) -> numpy.typing.NDArray:
        return self.energy_capability.is_self_consistent_stable

    @cached_property
    def structure_capability(self) -> StructureMetricsCapability:
        return self._get_capability(StructureMetricsCapability)

    @cached_property
    def energy_capability(self) -> EnergyMetricsCapability:
        return self._get_capability(EnergyMetricsCapability)

    @cached_property
    def property_capability(self) -> PropertyMetricsCapability:
        return self._get_capability(PropertyMetricsCapability)

    CapabilityT = TypeVar("CapabilityT", bound=BaseMetricsCapability)

    def _get_capability(self, capability: Type[CapabilityT]) -> CapabilityT:
        assert (
            capability in self.available_capability_types
        ), f"Capability {capability} is not available. Must be one of {self.available_capability_types}."
        return unique_item(cap for cap in self.capabilities if isinstance(cap, capability))

    def _get_metric(self, metric: Type[BaseMetric]) -> BaseMetric:
        assert (
            metric in self.available_metrics
        ), f"Metric {metric} is not available. Must be one of {self.available_metrics}."
        if metric not in self._metrics:
            capabilities: dict[str, BaseMetricsCapability | None] = {
                StructureMetricsCapability.name: None,
                EnergyMetricsCapability.name: None,
                PropertyMetricsCapability.name: None,
            }
            capabilities.update({capability.name: capability for capability in self.capabilities})
            self._metrics[metric] = metric(**capabilities)

        return self._metrics[metric]

    def compute_metric(self, metric: Type[BaseMetric]) -> float | int:
        """Compute a single metric."""
        return self._get_metric(metric).value

    def compute_metrics(
        self,
        metrics: Sequence[Type[BaseMetric]] | Literal["all"],
        save_as: str | os.PathLike | None = None,
        pretty_print: bool = False,
    ) -> dict[str, float | int]:
        """Computes metrics and returns them as a dictionary. Optionally, saves the dictionary to a file.

        Args:
            metrics: List of metrics to compute. If "all", all available metrics are computed.
            save_as: Path to save the dictionary. If None, the dictionary is not saved.
            pretty_print: If True, the dictionary is printed in a pretty format.
        """

        metrics_dict: dict[str, dict] = {}
        metrics_classes = self.available_metrics if metrics == "all" else metrics

        for metric_cls in metrics_classes:
            metric = self._get_metric(metric_cls)
            logger.info(f"Computing metric {metric.name}")
            metrics_dict[metric.name] = {"value": metric.value, "description": metric.description}

        if pretty_print:
            logger.info(
                json.dumps(
                    {
                        k: (round(v, 4) if isinstance(v, float) else v)
                        for (k, v) in metrics_dict.items()
                    },
                    indent=4,
                )
            )

        if save_as is not None:
            # Make sure that the directory exists
            save_as = Path(save_as).resolve()
            os.makedirs(save_as.parent, exist_ok=True)
            with open(save_as, "w") as f:
                json.dump(metrics_dict, f, indent=4)
            logger.info(f"Saved metrics to {save_as}")

        return {k: v["value"] for k, v in metrics_dict.items()}

    def compute_all_metrics(self) -> dict[str, float | int]:
        """Computes all available metrics."""
        return self.compute_metrics(self.available_metrics)

    def as_dataframe(
        self,
        metrics: Sequence[Type[BaseMetric]] | Literal["all"] | None = None,
        save_as: str | os.PathLike | None = None,
    ) -> DataFrame:
        """Return aggregate metrics as a pandas DataFrame, along with additional information from each available capability."""

        metrics = metrics or []
        metrics_classes = self.available_metrics if metrics == "all" else metrics

        data = {
            "entry": list(self.capabilities[0].dataset),
            **{
                metric.pre_aggregation_name: metric.pre_aggregation_values
                for metric in [self._get_metric(m) for m in metrics_classes]
                if isinstance(metric, BaseAggregateMetric)
            },
        }

        df = DataFrame(
            data=data,
            index=[e.entry_id for e in self.capabilities[0].dataset],
        )
        dfs = [df] + [cap.as_dataframe() for cap in self.capabilities]
        assert all([len(df) == len(d) for d in dfs]), "DataFrames do not have the same length."
        df = pd.concat(dfs, axis=1)

        if save_as is not None:
            # convert the dataframe to a dict first to allow serialization/deserialization
            # by monty.json.
            dumpfn(df.to_dict("list"), save_as)

        return df

    T = TypeVar("T")

    @staticmethod
    def filter(data: list[T], mask: numpy.typing.NDArray) -> list[T]:
        """Filters a list of data points based on a boolean mask."""
        assert len(data) == len(mask), "Data and mask must have the same length."
        return [x for x, m in zip(data, mask) if m]


def get_all_metrics_classes() -> list[Type[BaseMetric]]:
    """Returns all metrics classes, except for base classes."""
    clsmembers: list[list[tuple[str, Type]]] = [
        getmembers(module, isclass)
        for module in [energy_metrics, property_metrics, structure_metrics]
    ]
    metric_classes = [
        x[1]
        for clsmembers_in_module in clsmembers
        for x in clsmembers_in_module
        if issubclass(x[1], BaseMetric)
    ]

    return [m for m in metric_classes if not m.__name__.startswith("Base")]
