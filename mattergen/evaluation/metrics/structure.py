# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Sequence

import cachetools
import numpy as np
import numpy.typing
import smact
from pandas import DataFrame
from pymatgen.core.composition import Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.stats import wasserstein_distance
from smact.screening import pauling_test
from tqdm import tqdm

from mattergen.evaluation.metrics.core import BaseAggregateMetric, BaseMetric, BaseMetricsCapability
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils.dataset_matcher import (
    DisorderedDatasetUniquenessComputer,
    OrderedDatasetUniquenessComputer,
    get_dataset_matcher,
    matches_to_mask,
)
from mattergen.evaluation.utils.logging import logger
from mattergen.evaluation.utils.metrics_structure_summary import MetricsStructureSummary
from mattergen.evaluation.utils.structure_matcher import (
    DisorderedStructureMatcher,
    OrderedStructureMatcher,
)
from mattergen.evaluation.utils.symmetry_analysis import (
    DefaultSpaceGroupAnalyzer,
    DisorderedSpaceGroupAnalyzer,
)


def get_space_group(
    structure: Structure,
    space_group_analyzer_cls: type[SpacegroupAnalyzer] = DefaultSpaceGroupAnalyzer,
) -> str:
    try:
        return space_group_analyzer_cls(structure=structure).get_space_group_symbol()
    except TypeError:
        # space group analysis failed, most likely due to overlapping atoms
        return "P1"


def all_structures_are_ordered(structures: Sequence[Structure]) -> bool:
    """Check if all structures are ordered."""
    return all([s.is_ordered for s in structures])


class StructureMetricsCapability(BaseMetricsCapability):
    name: str = "structure_capability"

    """Capability for computing structure metrics.
    The `structure_matcher` class determines how uniqueness and novelty are computed.
        atoms that could substitute for each other (via the Hume-Rothery rules) and then using the default pymatgen structure matching.
    """

    def __init__(
        self,
        structure_summaries: list[MetricsStructureSummary],
        reference_dataset: ReferenceDataset,
        structure_matcher: OrderedStructureMatcher
        | DisorderedStructureMatcher,  # how are uniqueness and novelty computed
        n_failed_jobs: int = 0,
    ) -> None:
        super().__init__(structure_summaries=structure_summaries, n_failed_jobs=n_failed_jobs)
        _structures = [s.structure for s in structure_summaries]
        all_structures_ordered = (
            all_structures_are_ordered(_structures) and reference_dataset.is_ordered
        )
        if not all_structures_ordered:
            assert isinstance(structure_matcher, DisorderedStructureMatcher), (
                "If at least one structure is disordered, "
                "structure_matcher must be a DisorderedStructureMatcher."
            )
            logger.info(
                "At least one structure is disordered. Using DisorderedDatasetUniquenessComputer."
            )
        self.reference_dataset = reference_dataset  # note: not all metrics use this, so it could be a separate capability
        self.structure_matcher = structure_matcher
        self.ensure_reference_dataset_has_material_ids()
        self.uniqueness_computer: OrderedDatasetUniquenessComputer | DisorderedDatasetUniquenessComputer = (
            OrderedDatasetUniquenessComputer(structure_matcher)
            if all_structures_ordered
            else DisorderedDatasetUniquenessComputer(structure_matcher)
        )
        self.dataset_matcher = get_dataset_matcher(all_structures_ordered, structure_matcher)

    def ensure_reference_dataset_has_material_ids(self) -> None:
        """
        We're using material_ids to match structures between the reference dataset and the data.
        If the reference dataset doesn't have material_ids, we add them here and set them
        to the index of the entry in the reference dataset.
        """
        if (
            len(self.reference_dataset) > 0
            and next(iter(self.reference_dataset)).data.get("material_id") is None
        ):
            logger.warning(
                "Reference dataset does not have material_ids. Adding material_ids to reference dataset."
            )
            for i, entry in enumerate(self.reference_dataset):
                if "material_id" in entry.data:
                    raise ValueError(
                        "Found material_id in some entries of the reference dataset, but not all."
                        "Please ensure that either all entries have material_ids or none do."
                    )
                entry.data["material_id"] = i

    @property
    def structures(self) -> list[Structure]:
        return [s.structure for s in self._structure_summaries]

    @cached_property
    def chemistry_agnostic_structures(self) -> list[Structure]:
        chemistry_agnostic_structures = [deepcopy(s) for s in self.structures]
        for s in chemistry_agnostic_structures:
            s.replace_species({Element(k.name): Element("Cs") for k in list(set(s.species))})

        return chemistry_agnostic_structures

    @cached_property
    def is_unique(self) -> numpy.typing.NDArray[np.bool_]:
        """
        Returns a boolean mask of the same length as `data_entries` in which each item is True
        for the first structure from a set of duplicates and otherwise False.
        """
        return self.uniqueness_computer(self.dataset)

    @cached_property
    def is_novel(self) -> numpy.typing.NDArray[np.bool_]:
        """
        Returns a boolean mask of the same length as `data_entries` in which each item is True
        for structures that are not present in the reference dataset and otherwise False.
        """
        novelty_mask = np.logical_not(self.is_in_reference)
        return novelty_mask

    @cached_property
    def is_in_reference(self) -> numpy.typing.NDArray[np.bool_]:
        """
        Returns a boolean mask of the same length as `data_entries` in which each item is True
        for structures that are present in the reference dataset and otherwise False.
        """

        return matches_to_mask(self.matches_in_reference.keys(), len(self.dataset))

    @cached_property
    def matches_in_reference(self) -> dict[int, list[str]]:
        return self.dataset_matcher(self.dataset, self.reference_dataset)

    @cached_property
    def is_explored(self) -> numpy.typing.NDArray[np.bool_]:
        """Returns a mask of whether structures are in explored chemical systems (>1 entry in reference)."""
        return np.array(
            [
                structure.composition.chemical_system in self.reference_dataset.entries_by_chemsys
                for structure in self.structures
            ]
        )

    def as_dataframe(self) -> DataFrame:
        return DataFrame(
            data={
                # including is_unique and is_novel here might trigger an expensive computation
                "is_unique": self.is_unique,
                "is_novel": self.is_novel,
                "is_explored": self.is_explored,
            },
            index=[e.entry_id for e in self.dataset],
        )

    @cached_property
    def num_atoms(self) -> numpy.typing.NDArray[np.int_]:
        return np.array([len(structure) for structure in self.structures])

    @cached_property
    def space_group_symbols(self) -> list[str]:
        return [get_space_group(structure) for structure in self.structures]

    @cached_property
    def chemistry_agnostic_space_group_symbols(self) -> list[str]:
        return [get_space_group(structure) for structure in self.chemistry_agnostic_structures]

    @cached_property
    def substitution_aware_space_group_symbols(self) -> list[str]:
        """
        Returns a list of space group symbols for each structure in the dataset, once the
        structures have been modified to account for possible substitutions of atoms that
        could substitute for each other (via the Hume-Rothery rules).
        """
        return [
            get_space_group(structure, DisorderedSpaceGroupAnalyzer)
            for structure in self.structures
        ]

    # Ignore "desc" for the cache because it is irrelevant.
    @cachetools.cached(cache={}, key=lambda self, *args, **kwargs: self)
    def compute_num_matches(
        self,
        desc: str = "",
    ) -> float:
        """
        Returns the number of matches between the data and reference entries.
        """
        num_matches = len(self.is_novel) - sum(self.is_novel)
        return num_matches


# -----------------------------#
# Metrics
# -----------------------------#


@dataclass(frozen=True)
class BaseStructureMetric(BaseMetric):
    # Use for metrics that have access to structure data.
    required_capabilities = (StructureMetricsCapability,)

    @property
    def name(self) -> str:
        return "base_structure_metric"

    def __init__(
        self,
        structure_capability: StructureMetricsCapability,
        **kwargs,  # eat up unused kwargs (i.e., other capabilities)
    ):
        self.structure_capability = structure_capability
        self.reference_dataset = self.structure_capability.reference_dataset
        self.dataset = self.structure_capability.dataset


class FracUniqueSystems(BaseStructureMetric):
    name = "frac_unique_systems"

    @property
    def description(self) -> str:
        return "Fraction of structures in sampled data that have a unique chemical system within this set."

    @cached_property
    def value(self) -> float:
        # number of distinct chemical systems
        return len(
            set(
                structure.composition.chemical_system
                for structure in self.structure_capability.structures
            )
        ) / len(self.structure_capability.structures)


class Precision(BaseStructureMetric):
    name = "precision"

    @property
    def description(self) -> str:
        return f"Precision of structures in sampled data compared with {self.reference_dataset.name}. This is the fraction of structures in sampled data that have a matching structure in {self.reference_dataset.name}."

    @cached_property
    def value(self) -> float:
        """
        Returns the fraction of structures in self.data.data_structures that are present in
        self.reference_structures.
        """
        return self.structure_capability.is_in_reference.mean()


class Recall(BaseStructureMetric):
    name = "recall"

    @property
    def description(self) -> str:
        return f"Recall of structures in sampled data compared with structures in {self.reference_dataset.name}. This is the fraction of structures in sampled data that have a matching structure in {self.reference_dataset.name}."

    @cached_property
    def value(self) -> float:
        """
        Fraction of reference_structures that are in data_structures
        """
        match_dict = self.structure_capability.matches_in_reference
        ref_points_with_at_least_one_match = set([val for v in match_dict.values() for val in v])
        return len(ref_points_with_at_least_one_match) / len(self.reference_dataset)


class FracUniqueStructures(BaseStructureMetric, BaseAggregateMetric):
    aggregation_method: Literal["mean"] = "mean"
    name = "frac_unique_structures"
    pre_aggregation_name = "unique"

    @property
    def description(self) -> str:
        return "Fraction of unique structures in sampled data."

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        return self.structure_capability.is_unique


class FracNovelStructures(BaseStructureMetric, BaseAggregateMetric):
    aggregation_method: Literal["mean"] = "mean"
    name = "frac_novel_structures"
    pre_aggregation_name = "novel"

    @property
    def description(self) -> str:
        return "Fraction of novel structures in sampled data."

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        return self.structure_capability.is_novel


class FracNovelUniqueStructures(BaseStructureMetric, BaseAggregateMetric):
    aggregation_method: Literal["mean"] = "mean"
    name = "frac_novel_unique_structures"
    pre_aggregation_name = "novel_unique"

    @property
    def description(self) -> str:
        return "Fraction of novel unique structures in sampled data."

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        return self.structure_capability.is_novel & self.structure_capability.is_unique


class AvgStructureValidity(BaseStructureMetric, BaseAggregateMetric):
    aggregation_method: Literal["mean"] = "mean"
    name = "avg_structure_validity"
    pre_aggregation_name = "structure_validity"

    @property
    def description(self) -> str:
        return "Average structural validity of structures in sampled data. Any atom-atom distances less than 0.5 Angstroms or a volume less than 0.1 Angstrom**3 are considered invalid ."

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        return np.array(
            [
                structure_validity(structure=structure)
                for structure in tqdm(
                    self.structure_capability.structures, desc="Computing avg structure validity"
                )
            ]
        )


class AvgCompValidity(BaseStructureMetric, BaseAggregateMetric):
    aggregation_method: Literal["mean"] = "mean"
    name = "avg_comp_validity"
    pre_aggregation_name = "comp_validity"

    @property
    def description(self) -> str:
        return "Average composition validity (according to smact) of structures in sampled data."

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        return np.array(
            [
                is_smact_valid(structure=structure)
                for structure in tqdm(
                    self.structure_capability.structures, desc="Computing avg comp validity"
                )
            ]
        )


class AvgStructureCompValidity(BaseStructureMetric, BaseAggregateMetric):
    aggregation_method: Literal["mean"] = "mean"
    name = "avg_structure_comp_validity"
    pre_aggregation_name = "structure_comp_validity"

    @property
    def description(self) -> str:
        return "Average number of structures in sampled data that are both valid structures and have a valid smact compositions."

    def compute_pre_aggregation_values(self) -> numpy.typing.NDArray:
        valid_comp = [
            structure_validity(structure=structure)
            for structure in self.structure_capability.structures
        ]
        valid_struct = [
            is_smact_valid(structure=structure)
            for structure in self.structure_capability.structures
        ]
        return np.array(valid_comp) & np.array(valid_struct)


class FracNovelSystems(BaseStructureMetric):
    name = "frac_novel_systems"

    @property
    def description(self) -> str:
        return f"Fraction of distinct chemical systems in sampled data and not in {self.reference_dataset.name}."

    @cached_property
    def value(self) -> float:
        chemical_systems = set(
            [
                structure.composition.chemical_system
                for structure in self.structure_capability.structures
            ]
        )
        return len(
            [
                chemsys
                for chemsys in chemical_systems
                if chemsys not in self.reference_dataset.entries_by_chemsys
            ]
        ) / len(self.structure_capability.structures)


# -----------------------------#
# Utility functions
# -----------------------------#


def is_smact_valid(structure: Structure) -> bool:
    """
    Returns True if the structure is valid according to the
    smact validity checker else False.
    """
    elem_counter = Counter(structure.atomic_numbers)
    composition = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
    elems, counts = list(zip(*composition))
    counts = np.array(counts)
    counts = counts / np.gcd.reduce(counts)
    comps: tuple[int, ...] = tuple(np.array(counts).astype("int"))
    try:
        return smact_validity(comp=elems, count=comps, use_pauling_test=True, include_alloys=True)
    except TypeError:
        raise TypeError(
            f"SMACT validity checker failed. Check that all elements {structure.composition} present in the structure are also present in smact.element_dictionary()."
        )
    # HOTFIX: decode error sometimes occurrs the first time the smact_validity function is called, but not after that
    except UnicodeDecodeError:
        return smact_validity(comp=elems, count=comps, use_pauling_test=True, include_alloys=True)


def smact_validity(
    comp: tuple[int, ...] | tuple[str, ...],
    count: tuple[int, ...],
    use_pauling_test: bool = True,
    include_alloys: bool = True,
    include_cutoff: bool = False,
    use_element_symbol: bool = False,
) -> bool:
    """Computes SMACT validity.

    Args:
        comp: Tuple of atomic number or element names of elements in a crystal.
        count: Tuple of counts of elements in a crystal.
        use_pauling_test: Whether to use electronegativity test. That is, at least in one
            combination of oxidation states, the more positive the oxidation state of a site,
            the lower the electronegativity of the element for all pairs of sites.
        include_alloys: if True, returns True without checking charge balance or electronegativity
            if the crystal is an alloy (consisting only of metals) (default: True).
        include_cutoff: assumes valid crystal if the combination of oxidation states is more
            than 10^6 (default: False).

    Returns:
        True if the crystal is valid, False otherwise.
    """
    assert len(comp) == len(count)
    if use_element_symbol:
        elem_symbols = comp
    else:
        elem_symbols = tuple([str(Element.from_Z(Z=elem)) for elem in comp])  # type:ignore
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    n_comb = np.prod([len(ls) for ls in ox_combos])
    # If the number of possible combinations is big, it'd take too much time to run the smact checker
    # In this case, we assume that at least one of the combinations is valid
    if n_comb > 1e6 and include_cutoff:
        return True
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        return False


def structure_validity(structure: Structure, cutoff: float = 0.5) -> bool:
    dist_mat = structure.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
    # Note: the threshold 0.1 comes from the CDVAE code
    # https://github.com/txie-93/cdvae/blob/f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea/scripts/eval_utils.py#L170
    if dist_mat.min() < cutoff or structure.volume < 0.1:
        return False
    else:
        return True
