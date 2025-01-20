# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from itertools import combinations

import numpy as np
from emmet.core.utils import get_sg
from pymatgen.analysis.structure_matcher import (
    AbstractComparator,
    OrderDisorderElementComparator,
    StructureMatcher,
)
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure

from mattergen.evaluation.utils.globals import MAX_RMSD


class RMSDStructureMatcher(StructureMatcher):
    """
    Structure matcher used for computing RMSD distance between structures. Has looser
    tolerances than the default pymatgen StructureMatcher to ensure that we can get an
    atom alignment even in case structures don't match.
    """

    def __init__(
        self,
    ):
        super().__init__(
            ltol=0.5,
            stol=MAX_RMSD,
            angle_tol=10,
            primitive_cell=True,
            scale=False,
            attempt_supercell=True,
            allow_subset=False,
        )


class OrderedStructureMatcher(StructureMatcher):
    def __init__(
        self,
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5,
        primitive_cell: float = True,
        scale: float = True,
        attempt_supercell: float = False,
        allow_subset: float = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol,
            primitive_cell=primitive_cell,
            scale=scale,
            attempt_supercell=attempt_supercell,
            allow_subset=allow_subset,
            *args,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "OrderedStructureMatcher"


class DefaultOrderedStructureMatcher(OrderedStructureMatcher):
    """
    Ordered structure matcher with default parameters. No args or kwargs are passed in order
    to ensure consistent behavior across all instances.
    """

    def __init__(self):
        super().__init__()


class DisorderedStructureMatcher(StructureMatcher):
    def __init__(
        self,
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5.0,
        primitive_cell: bool = True,
        scale: bool = True,
        comparator: AbstractComparator = OrderDisorderElementComparator(),
        attempt_supercell: bool = True,
        allow_subset: bool = True,
        relative_radius_difference_threshold: float = 0.3,
        electronegativity_difference_threshold: float = 1.0,
        reduced_formula_atol: float = 1e-2,  # 1e-8 is the default value in pymatgen composition.almost_equals()
        reduced_formula_rtol: float = 1e-1,  # 1e-1 is the default value in pymatgen composition.almost_equals()
        *args,
        **kwargs,
    ):
        super().__init__(
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol,
            primitive_cell=primitive_cell,
            allow_subset=allow_subset,
            attempt_supercell=attempt_supercell,
            scale=scale,
            comparator=comparator,
            *args,
            **kwargs,
        )
        self.relative_radius_difference_threshold = relative_radius_difference_threshold
        self.electronegativity_difference_threshold = electronegativity_difference_threshold
        self.ordered_structurematcher = OrderedStructureMatcher(
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol,
            primitive_cell=primitive_cell,
            scale=scale,
        )
        self.reduced_formula_atol = reduced_formula_atol
        self.reduced_formula_rtol = reduced_formula_rtol

    @property
    def name(self) -> str:
        return "DisorderedStructureMatcher"

    def fit(self, structure_1: Structure, structure_2: Structure) -> bool:
        """
        Returns True if the structures are equivalent, False otherwise.
        First checks whether the composition of the structures is similar.
        Then checks whether the structures are ordered or disordered.
        If both structures are ordered, they are first compared directly,
        and if they do not match, one of the structures is disordered and compared again.
        If one of the structures is disordered, the disordered comparer is used directly.
        The structures are first copied and their oxidation states are removed.
        """
        structure_1_nooxi = structure_1.copy().remove_oxidation_states()
        structure_2_nooxi = structure_2.copy().remove_oxidation_states()

        if structure_1_nooxi == structure_2_nooxi:
            return True

        if structure_1_nooxi.is_ordered and structure_2_nooxi.is_ordered:
            # Strict comparison of reduced formulas if one of the structures is disordered
            if (
                structure_2_nooxi.composition.reduced_formula
                != structure_1_nooxi.composition.reduced_formula
            ):
                return False
            # First do the simple comparison, and exit the loop only if a match is found
            if get_sg(structure_1_nooxi) == get_sg(structure_2_nooxi):
                if self.ordered_structurematcher.fit(structure_1_nooxi, structure_2_nooxi):
                    return True

            # Then disorder one of the structures and compare again using the disordered matcher
            structure_1_nooxi, can_be_disordered_1 = try_make_structure_disordered(
                structure=structure_1_nooxi,
                relative_radius_difference_threshold=self.relative_radius_difference_threshold,
                electronegativity_difference_threshold=self.electronegativity_difference_threshold,
            )
            if can_be_disordered_1:
                return super().fit(structure_1_nooxi, structure_2_nooxi)
            return False

        # If at least one of the structures is already disordered, use the disordered matcher
        # Loose comparison of reduced formulas if one of the structures is disordered
        # We need to use fractional_composition here because otherwise the match fails for
        # structures with different number of sites (e.g., supercells)
        if not structure_1_nooxi.composition.fractional_composition.almost_equals(
            structure_2_nooxi.composition.fractional_composition,
            atol=self.reduced_formula_atol,
            rtol=self.reduced_formula_rtol,
        ):
            return False
        return super().fit(structure_1_nooxi, structure_2_nooxi)


class DefaultDisorderedStructureMatcher(DisorderedStructureMatcher):
    """
    Disordered structure matcher with default parameters. No args or kwargs are passed in order
    to ensure consistent behavior across all instances.
    """

    def __init__(self):
        super().__init__()


def get_cliques_out_of_list_of_pairs(pairs: list[list[Element]]) -> list[list[Element]]:
    # create cliques of elements that can substitute each other
    cliques: list[list[Element]] = [[]]
    for pair in pairs:  # Loop over pairs of elements that can substitute each other
        previously_appended_to_group = None
        for i, group in enumerate(cliques):
            # If any of the elements in the pair are already in the group, there are two options
            if pair[0] in group or pair[1] in group:
                if previously_appended_to_group is not None:
                    # Now, if this pair is matching this group but was already matched to another group,
                    # append the new group to the old group and empty the new group
                    cliques[previously_appended_to_group].extend(group)
                    cliques[i] = []
                else:
                    # If instead this is the first group that the pair is matched to, append the pair to the group
                    # and mark this group as the one that the pair was matched to
                    cliques[i].extend(pair)
                    previously_appended_to_group = i
        # If by the end of the loop the pair was not matched to any group, create a new group
        if previously_appended_to_group is None:
            cliques.append(pair)

    # Remove empty groups and duplicates within groups
    return [list(set(group)) for group in cliques if len(group) > 0]


def make_structure_disordered(structure: Structure, substitution: list[list[Element]]) -> Structure:
    """
    Returns a copy of the structure where the cliques of elements that can substitute each other are replaced by partial occupancies.
    The partial occupancies are calculated based on the atomic fractions of the elements in the clique.
    """
    disordered_structure = structure.copy().remove_oxidation_states()
    atomic_fractions = {
        str(species): disordered_structure.composition.get_atomic_fraction(str(species))
        for species in list(disordered_structure.composition)
    }
    for substitution_clique in substitution:
        these_atomic_fractions = {
            species: atomic_fractions[str(species)] for species in substitution_clique
        }
        total_atomic_fraction = sum(these_atomic_fractions.values())
        these_atomic_fractions = {
            species: atomic_fraction / total_atomic_fraction
            for species, atomic_fraction in these_atomic_fractions.items()
        }
        disordered_structure.replace_species(
            {
                str(species): "".join(
                    [
                        str(species) + str(these_atomic_fractions[species])
                        for species in substitution_clique
                    ]
                )
                for species in substitution_clique
            }
        )
    return disordered_structure


def do_elements_substitute(
    element_1: Element,
    element_2: Element,
    relative_radius_difference_threshold: float = 0.3,
    electronegativity_difference_threshold: float = 1.0,
) -> bool:
    """
    Returns whether two elements could substitute based on their atomic radius and electronegativity.
    This is a modified Hume-Rothery rule, where the relative atomic radius difference and the electronegativity difference
    thresholds are obtained from an analysis carried out on ICSD data.
    See the revised MatterGen paper for more details.
    """
    relative_atomic_radius_difference = abs(
        element_1.atomic_radius - element_2.atomic_radius
    ) / np.mean([element_1.atomic_radius, element_2.atomic_radius])
    electronegativity_difference = abs(element_1.X - element_2.X)
    return (
        relative_atomic_radius_difference <= relative_radius_difference_threshold
        and electronegativity_difference <= electronegativity_difference_threshold
    )


def check_is_disordered(
    structure: Structure,
    relative_radius_difference_threshold: float = 0.3,
    electronegativity_difference_threshold: float = 1.0,
) -> tuple[bool, list[list[Element]]]:
    """
    Function to estimate whether a structure can be thought as an ordered approximation of an alloy.
    Returns:

    is_disordered: can the structure be thought of as an alloy?
    substitutional_groups: list of sets of elements that could substitute for each other

    """
    structure_copy = structure.copy().remove_oxidation_states()

    substitutional_pairs = []
    for element_1, element_2 in combinations(list(structure_copy.composition), 2):
        # reduce to string of element
        if do_elements_substitute(
            element_1=element_1,
            element_2=element_2,
            relative_radius_difference_threshold=relative_radius_difference_threshold,
            electronegativity_difference_threshold=electronegativity_difference_threshold,
        ):
            substitutional_pairs.append([element_1, element_2])

    if len(substitutional_pairs) == 0:
        return False, [[]]

    substitutional_groups = get_cliques_out_of_list_of_pairs(pairs=substitutional_pairs)
    return True, substitutional_groups


def try_make_structure_disordered(
    structure: Structure,
    relative_radius_difference_threshold: float = 0.3,
    electronegativity_difference_threshold: float = 1.0,
) -> tuple[Structure, bool]:
    can_be_disordered, substitution_species = check_is_disordered(
        structure=structure,
        relative_radius_difference_threshold=relative_radius_difference_threshold,
        electronegativity_difference_threshold=electronegativity_difference_threshold,
    )
    return (
        make_structure_disordered(structure, substitution_species)
        if can_be_disordered
        else structure,
        can_be_disordered,
    )
