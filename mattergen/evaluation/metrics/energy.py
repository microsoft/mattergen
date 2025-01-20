# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Literal

import numpy as np
import numpy.typing
from pandas import DataFrame
from pymatgen.analysis.phase_diagram import PhaseDiagram
from tqdm import tqdm

from mattergen.evaluation.metrics.core import BaseAggregateMetric, BaseMetric, BaseMetricsCapability
from mattergen.evaluation.metrics.structure import StructureMetricsCapability
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils.globals import DEFAULT_STABILITY_THRESHOLD
from mattergen.evaluation.utils.logging import logger
from mattergen.evaluation.utils.metrics_structure_summary import MetricsStructureSummary
from mattergen.evaluation.utils.utils import expand_into_subsystems

# -----------------------------#
# Capabilities
# -----------------------------#


class MissingTerminalsError(ValueError):
    pass


def get_set_of_all_elements(structure_summaries: list[MetricsStructureSummary]) -> set[str]:
    """Returns a set of terminal chemical systems in the dataset."""
    return set(
        str(element) for x in structure_summaries for element in x.entry.composition.elements
    )


@dataclass(frozen=True)
class MissingTerminalsAndEnergy:
    """
    Class to store information about missing terminal systems and energy data in the reference dataset.
    """

    missing_terminals: list[str]
    missing_energy: list[str]

    @classmethod
    def from_dataset_and_reference(
        cls,
        structure_summaries: list[MetricsStructureSummary],
        reference: ReferenceDataset,
    ) -> "MissingTerminalsAndEnergy":
        terminal_systems = get_set_of_all_elements(structure_summaries)
        missing_terminals = list(terminal_systems - set(reference.entries_by_chemsys.keys()))
        # among non-missing terminal systems, check if any of them have missing energy data
        terminals_in_reference = terminal_systems & set(reference.entries_by_chemsys.keys())
        missing_energy = [
            chemsys
            for chemsys in terminals_in_reference
            if all([np.isnan(e.energy) for e in reference.entries_by_chemsys[chemsys]])
        ]
        return cls(missing_terminals=missing_terminals, missing_energy=missing_energy)

    @property
    def has_missing_terminals(self) -> bool:
        return len(self.missing_terminals) > 0

    @property
    def has_missing_energy(self) -> bool:
        return len(self.missing_energy) > 0

    @property
    def has_missing_data(self) -> bool:
        return self.has_missing_terminals or self.has_missing_energy


class EnergyMetricsCapability(BaseMetricsCapability):
    name: str = "energy_capability"
    missing_terminals_error_str = "Reference dataset does not contain sufficient data to compute energy metrics for the given dataset."

    """Capability for computing structure metrics."""

    @classmethod
    def check_missing_reference_terminal_systems(
        cls,
        structure_summaries: list[MetricsStructureSummary],
        reference_dataset: ReferenceDataset,
    ) -> MissingTerminalsAndEnergy:
        return MissingTerminalsAndEnergy.from_dataset_and_reference(
            structure_summaries=structure_summaries,
            reference=reference_dataset,
        )

    @classmethod
    def warn_missing_data(cls, missing_terminals: MissingTerminalsAndEnergy) -> None:
        logger.warning(cls.missing_terminals_error_str)
        if missing_terminals.has_missing_terminals:
            logger.warning(f"Missing terminal systems: {missing_terminals.missing_terminals}")
        if missing_terminals.has_missing_energy:
            logger.warning(
                f"Missing energy data for terminal systems: {missing_terminals.missing_energy}"
            )

    def __init__(
        self,
        structure_summaries: list[MetricsStructureSummary],
        reference_dataset: ReferenceDataset,
        stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
        n_failed_jobs: int = 0,
    ) -> None:
        if (
            missing_terminals := self.check_missing_reference_terminal_systems(
                structure_summaries, reference_dataset
            )
        ).has_missing_data:
            self.warn_missing_data(missing_terminals)
            raise MissingTerminalsError(self.missing_terminals_error_str)
        super().__init__(structure_summaries=structure_summaries, n_failed_jobs=n_failed_jobs)
        self.reference_dataset = reference_dataset
        self.stability_threshold = stability_threshold

    @property
    def is_stable(self) -> numpy.typing.NDArray[np.bool_]:
        """
        Returns a boolean mask of the same length as data_entries
        indicating whether each entry is stable or not.
        """
        return self.energy_above_hull <= self.stability_threshold

    @property
    def is_self_consistent_stable(self) -> numpy.typing.NDArray[np.bool_]:
        """
        Returns a boolean mask of the same length as data_entries
        indicating whether each entry is self-consistently stable or not.
        """
        return self.self_consistent_energy_above_hull <= self.stability_threshold

    @cached_property
    def energy_above_hull(self) -> numpy.typing.NDArray:
        """Returns energy above hull (eV) per atom with respect to the reference dataset."""
        result = np.zeros(len(self.dataset))
        for chemsys, entries in tqdm(
            self.dataset.entries_by_chemsys.items(),
            desc="Computing energies above hull",
        ):
            result[[e.entry_id for e in entries]] = np.array(
                self._get_energy_above_hull_per_atom_chemsys(chemsys)
            )
        return result

    @cached_property
    def self_consistent_energy_above_hull(self) -> numpy.typing.NDArray:
        """Returns the energy above hull (eV) per atom with respect to the convex hull that
        combines the reference dataset and the samples."""
        result = np.zeros(len(self.dataset))
        for chemsys, entries in tqdm(
            self.dataset.entries_by_chemsys.items(),
            desc="Computing self-consistent energies above hull",
        ):
            result[[e.entry_id for e in entries]] = np.array(
                self._get_self_consistent_energy_above_hull_per_atom_chemsys(chemsys)
            )
        return result

    def as_dataframe(self) -> DataFrame:
        return DataFrame(
            data={
                "energy_above_hull": self.energy_above_hull,
                "self_consistent_energy_above_hull": self.self_consistent_energy_above_hull,
            },
            index=[e.entry_id for e in self.dataset],
        )

    # ---------------------------------------------#
    # Helper functions shared by multiple metrics  #
    # ---------------------------------------------#

    def _get_phase_diagram(self, chemical_system: str) -> PhaseDiagram:
        """Returns the phase diagram for a given chemical system."""
        subsys = expand_into_subsystems(chemical_system)
        reference_entries = [
            entry
            for s in subsys
            for key in ["-".join(sorted(s))]
            for entry in self.reference_dataset.entries_by_chemsys.get(key, [])
            if not np.isnan(
                entry.energy
            )  # skip disordered structures, which have nan energy currently
        ]
        assert len(reference_entries) > 0, f"No reference data for {chemical_system}."
        return PhaseDiagram(reference_entries)

    @lru_cache
    def _get_energy_above_hull_per_atom_chemsys(self, chemsys: str) -> list[float]:
        """Returns a list of energies above hull per atom for a given chemical system."""
        phase_diagram = self._get_phase_diagram(chemsys)
        e_above_hull = [
            phase_diagram.get_e_above_hull(entry=e, allow_negative=True)
            for e in self.dataset.entries_by_chemsys[chemsys]
        ]
        for e, ehull in zip(self.dataset.entries_by_chemsys[chemsys], e_above_hull):
            logger.debug(
                f"{e.composition.reduced_formula}: energy above hull {ehull} (threshold {self.stability_threshold})"
            )
        return e_above_hull

    def _get_self_consistent_phase_diagram(self, chemical_system: str) -> PhaseDiagram:
        """Returns the internal phase diagram for a given chemical system.
        This is comprised of all reference entries that do not exactly match the chemical system, and
        of all entries belonging to the chemical system."""
        subsys = expand_into_subsystems(chemical_system)
        reference_entries = [
            entry
            for s in subsys
            for key in ["-".join(sorted(s))]
            for entry in self.reference_dataset.entries_by_chemsys.get(key, [])
            if key != chemical_system  # Do not get reference entries for the chemical system itself
            and not np.isnan(
                entry.energy
            )  # skip disordered structures, which have nan energy currently
        ]
        reference_entries += self.dataset.entries_by_chemsys.get(chemical_system, [])
        assert len(reference_entries) > 0, f"No data for {chemical_system}."
        return PhaseDiagram(reference_entries)

    def _get_full_phase_diagram(self, chemical_system: str) -> PhaseDiagram:
        """Returns the total phase diagram for a given chemical system.
        This is comprised of all reference entries  and
        of all entries belonging to the chemical system."""
        subsys = expand_into_subsystems(chemical_system)
        reference_entries = [
            entry
            for s in subsys
            for key in ["-".join(sorted(s))]
            for entry in self.reference_dataset.entries_by_chemsys.get(key, [])
            if not np.isnan(
                entry.energy
            )  # skip disordered structures, which have nan energy currently
        ]
        reference_entries += self.dataset.entries_by_chemsys.get(chemical_system, [])
        assert len(reference_entries) > 0, f"No data for {chemical_system}."
        return PhaseDiagram(reference_entries)

    @lru_cache
    def _get_self_consistent_energy_above_hull_per_atom_chemsys(self, chemsys: str) -> list[float]:
        """Returns a list of self-consistent energies above hull per atom for a given chemical system."""
        phase_diagram = self._get_self_consistent_phase_diagram(chemsys)
        e_above_hull = [
            phase_diagram.get_e_above_hull(entry=e, allow_negative=True)
            for e in self.dataset.entries_by_chemsys[chemsys]
        ]
        return e_above_hull


# -----------------------------#
# Metrics
# -----------------------------#


@dataclass(frozen=True)
class BaseEnergyMetric(BaseMetric):
    # Use for metrics that have access to structure and energy data.
    # In principle, we could have two classes, one for energy-only capabilities and one for structure+energy capabilities;
    # however, since the input data already contains both structure and energy data, we can use a single class for both.
    required_capabilities = (StructureMetricsCapability, EnergyMetricsCapability)

    @property
    def name(self) -> str:
        return "base_energy_metric"

    def __init__(
        self,
        structure_capability: StructureMetricsCapability,
        energy_capability: EnergyMetricsCapability,
        **kwargs,  # eat up unused kwargs (i.e., other capabilities)
    ):
        self.structure_capability = structure_capability
        self.energy_capability = energy_capability
        self.reference_dataset = self.energy_capability.reference_dataset


class FracSuccessfulJobs(BaseEnergyMetric):
    name = "frac_successful_jobs"

    @property
    def description(self) -> str:
        return "Fraction of structures whose jobs ran successfully."

    @cached_property
    def value(self) -> float:
        return (
            len(self.energy_capability._structure_summaries) / self.energy_capability.total_submitted_jobs
        )


class AvgRMSDFromRelaxation(BaseEnergyMetric, BaseAggregateMetric):
    aggregation_method: Literal["nanmean"] = "nanmean"
    name = "avg_rmsd_from_relaxation"
    pre_aggregation_name = "rmsd_from_relaxation"

    @property
    def description(self) -> str:
        return "root mean square displacements of atoms (Angstrom) from initial to final DFT relaxation steps in sampled data."

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        return np.array([d.rmsd_from_relaxation for d in self.energy_capability._structure_summaries])


class AvgEnergyAboveHullPerAtom(BaseEnergyMetric, BaseAggregateMetric):
    aggregation_method: Literal["mean"] = "mean"
    name = "avg_energy_above_hull_per_atom"
    pre_aggregation_name = "energy_above_hull_per_atom"

    @property
    def description(self) -> str:
        return "Average energy above hull per atom (eV/atom) of structures in sampled data."

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        return self.energy_capability.energy_above_hull


class FracStableStructures(BaseEnergyMetric, BaseAggregateMetric):
    name = "frac_stable_structures"
    pre_aggregation_name = "stable"

    @property
    def description(self) -> str:
        return f"Fraction of stable structures in sampled data within {self.energy_capability.stability_threshold} (eV/atom) above convex hull of {self.reference_dataset.name}."

    @cached_property
    def value(self) -> float:
        return self.pre_aggregation_values.sum() / self.energy_capability.total_submitted_jobs

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        return self.energy_capability.is_stable


class FracNovelUniqueStableStructures(BaseEnergyMetric, BaseAggregateMetric):
    name = "frac_novel_unique_stable_structures"
    pre_aggregation_name = "novel_unique_stable"

    @property
    def description(self) -> str:
        return (
            f"Fraction of novel unique stable structures in sampled data within {self.energy_capability.stability_threshold} (eV/atom) "
            + f"above convex hull of {self.reference_dataset.name}."
        )

    @cached_property
    def value(self) -> float:
        return self.pre_aggregation_values.sum() / self.energy_capability.total_submitted_jobs

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        return (
            self.structure_capability.is_novel
            & self.structure_capability.is_unique
            & self.energy_capability.is_stable
        )
