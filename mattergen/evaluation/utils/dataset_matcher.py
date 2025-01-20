# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
from typing import Iterable, List, Mapping

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from tqdm import tqdm

from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils.logging import logger
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DisorderedStructureMatcher,
    OrderedStructureMatcher,
)


def get_matches(
    structure_matcher: StructureMatcher, d1: List[Structure], d2: List[Structure]
) -> dict[int, list[int]]:
    """
    Iterates d1 to find matches in d2.

    Args:
        structure_matcher: StructureMatcher to use for comparison.
        d1: List of structures to compare.
        d2: List of structures to compare against.

    Returns:
        matches: Dictionary of matches. Key is the index of the structure in d1 and value is the index of the structure in d2.

    """
    matches: dict[int, list[int]] = defaultdict(list)

    for i in range(len(d1)):
        for j in range(len(d2)):
            if structure_matcher.fit(d1[i], d2[j]):
                matches[i].append(j)
    return matches


def get_unique(structure_matcher: StructureMatcher, structures: List[Structure]) -> List[int]:

    if len(structures) == 1:
        return [0]

    unique_structures: list[Structure] = []
    unique_idx: list[int] = []
    for idx, structure in enumerate(structures):
        unique = True
        for structure_2 in unique_structures:
            if structure_matcher.fit(structure, structure_2):
                unique = False
                break
        if unique:
            unique_structures.append(structure)
            unique_idx.append(idx)

    return unique_idx


def get_dataset_matcher(
    all_structures_ordered: bool, structure_matcher: StructureMatcher
) -> "DatasetMatcher":
    if all_structures_ordered:
        return OrderedDatasetMatcher(structure_matcher)
    return DisorderedDatasetMatcher(structure_matcher)


def get_global_index_from_local_index(
    entries_mapping_by_key: Mapping[str, list[ComputedStructureEntry]],
    local_index: Mapping[str, list[int]],
) -> list[int]:
    """Turn local structure chemsys index into global structure mask."""
    global_indices = [
        entries_mapping_by_key[k][vv].entry_id for k, v in local_index.items() for vv in v
    ]
    return global_indices


def get_global_match_dict_from_local_dict(
    data_entries_mapping_by_key: Mapping[str, list[ComputedStructureEntry]],
    reference_entries_mapping_by_key: Mapping[str, list[ComputedStructureEntry]],
    local_index: Mapping[str, dict[int, list[int]]],
) -> dict[int, list[str]]:
    global_match_dict = {}
    for k, match_dict in local_index.items():
        if len(match_dict) == 0 or max(len(v) for v in match_dict.values()) == 0:
            continue
        # Get the mapping of the data and reference entries only once, as it requires disk access
        data_entries_mapping = data_entries_mapping_by_key[k]
        reference_entries_mapping = reference_entries_mapping_by_key[k]
        for d1_ix, ref_ix_list in match_dict.items():
            global_match_dict[data_entries_mapping[d1_ix].entry_id] = [
                reference_entries_mapping[match_ix].data["material_id"] for match_ix in ref_ix_list
            ]
    return global_match_dict


def get_mask_from_local_index(
    entries_mapping_by_key: Mapping[str, list[ComputedStructureEntry]],
    local_index: Mapping[str, List[int]],
) -> np.typing.NDArray[np.bool_]:
    """Turn local structure chemsys index into global structure mask."""
    global_indices = get_global_index_from_local_index(entries_mapping_by_key, local_index)
    total_num_entries = sum(len(v) for v in entries_mapping_by_key.values())
    mask = np.zeros(total_num_entries, dtype=bool)
    mask[global_indices] = True
    return mask


class OrderedDatasetUniquenessComputer:
    def __init__(self, structure_matcher: StructureMatcher = DefaultDisorderedStructureMatcher()):
        self.structure_matcher = structure_matcher

    def __call__(self, dataset: ReferenceDataset) -> np.typing.NDArray[bool]:
        local_index: dict[str, List[int]] = {}
        for reduced_formula, data_entries in tqdm(
            dataset.entries_by_reduced_formula.items(),
            desc="Finding unique structures by reduced formula",
        ):
            structures = [e.structure for e in data_entries]
            assert all(
                [s.is_ordered for s in structures]
            ), "OrderedDatasetUniquenessComputer only works for ordered structures."
            local_index[reduced_formula] = get_unique(self.structure_matcher, structures)

        return get_mask_from_local_index(dataset.entries_by_reduced_formula, local_index)


class DisorderedDatasetUniquenessComputer:
    def __init__(self, structure_matcher: StructureMatcher = DefaultDisorderedStructureMatcher()):
        self.structure_matcher = structure_matcher

    def __call__(self, dataset: "ReferenceDataset") -> np.typing.NDArray[bool]:
        local_index: dict[str, List[int]] = {}
        for chemsys, data_entries in tqdm(
            dataset.entries_by_chemsys.items(),
            desc="Finding unique structures by chemsys",
        ):
            structures = [e.structure for e in data_entries]
            if not all([s.is_ordered for s in structures]):
                logger.warning(
                    "Using DisorderedDatasetUniquenessComputer for ordered structures. "
                    "This is less efficient than using OrderedDatasetUniquenessComputer."
                )
            local_index[chemsys] = get_unique(self.structure_matcher, structures)

        return get_mask_from_local_index(dataset.entries_by_chemsys, local_index)


def matches_to_mask(match_idx: Iterable[int], num_samples: int) -> np.typing.NDArray[bool]:
    """
    Convert matches to a boolean mask.

    Args:
        match_idx: List of indices of the structures from the input dataset which have a match
            in the reference dataset.
        num_samples: Number of structures in the input dataset.

    Returns:
        mask: Boolean mask of length num_samples. True if the structure has a match, False if not.
    """
    mask = np.zeros(num_samples, dtype=bool)
    mask[list(match_idx)] = True
    return mask


class DatasetMatcher:
    """
    Class to match a dataset of structures to a reference dataset.
    Can be used to compute novelty of the input dataset w.r.t. the reference dataset or
    to compute the recall.
    """

    def __init__(
        self, structure_matcher: OrderedStructureMatcher | DisorderedStructureMatcher
    ) -> None:
        self.structure_matcher = structure_matcher

    def grouped_dataset_entries(
        self, dataset: ReferenceDataset
    ) -> Mapping[str, list[ComputedStructureEntry]]:
        """
        Returns a dictionary of entries grouped by a key, e.g., chemsys or reduced_formula.
        To be implemented by the concrete dataset matcher.
        """
        raise NotImplementedError

    def __call__(
        self, dataset: ReferenceDataset, reference_dataset: ReferenceDataset
    ) -> dict[int, list[str]]:
        """
        For each entry in the dataset, check if there is a match in the reference dataset.

        Args:
            dataset: Dataset to match.
            reference_dataset: Reference dataset to match against.

        Returns:
            global_match_idx: Dictionary of matches. Key is the index of the structure in the input dataset and
              value is a list of the material_ids (str) of the matching structures in the reference dataset
        """
        local_match_indices: dict[str, dict[int, list[int]]] = {}
        grouped_dataset_entries = self.grouped_dataset_entries(dataset=dataset)
        grouped_reference_entries = self.grouped_dataset_entries(dataset=reference_dataset)
        for group_key, data_entries in tqdm(
            grouped_dataset_entries.items(),
            desc="Finding novel structures",
        ):
            data_structures = [e.structure for e in data_entries]
            reference_structures = [
                e.structure for e in grouped_reference_entries.get(group_key, [])
            ]
            matches = get_matches(
                self.structure_matcher,
                data_structures,
                reference_structures,
            )
            local_match_indices[group_key] = matches

        global_match_dict = get_global_match_dict_from_local_dict(
            grouped_dataset_entries, grouped_reference_entries, local_match_indices
        )

        return global_match_dict


class OrderedDatasetMatcher(DatasetMatcher):
    def __init__(self, structure_matcher: OrderedStructureMatcher):
        super().__init__(structure_matcher=structure_matcher)

    def grouped_dataset_entries(
        self, dataset: ReferenceDataset
    ) -> Mapping[str, list[ComputedStructureEntry]]:
        """
        Ordered dataset matcher groups by reduced formula.
        """
        return dataset.entries_by_reduced_formula


class DisorderedDatasetMatcher(DatasetMatcher):
    def __init__(self, structure_matcher: DisorderedStructureMatcher):
        super().__init__(structure_matcher=structure_matcher)

    def grouped_dataset_entries(
        self, dataset: ReferenceDataset
    ) -> Mapping[str, list[ComputedStructureEntry]]:
        """
        Disordered dataset matcher groups by chemsys.
        """
        return dataset.entries_by_chemsys
