# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from mattergen.evaluation.utils.structure_matcher import try_make_structure_disordered


class DefaultSpaceGroupAnalyzer(SpacegroupAnalyzer):
    def __init__(
        self,
        structure: Structure,
    ):
        super().__init__(structure, symprec=0.1, angle_tolerance=5.0)


class DisorderedSpaceGroupAnalyzer(SpacegroupAnalyzer):
    def __init__(
        self,
        structure: Structure,
    ):
        structure, _ = try_make_structure_disordered(
            structure=structure,
            relative_radius_difference_threshold=0.3,
            electronegativity_difference_threshold=1.0,
        )
        super().__init__(structure, symprec=0.1, angle_tolerance=5.0)


class StrictSpaceGroupAnalyzer(SpacegroupAnalyzer):
    def __init__(
        self,
        structure: Structure,
    ):
        super().__init__(structure, symprec=0.01, angle_tolerance=5.0)


class DisorderedStrictSpaceGroupAnalyzer(SpacegroupAnalyzer):
    def __init__(
        self,
        structure: Structure,
    ):
        structure, _ = try_make_structure_disordered(
            structure=structure,
            relative_radius_difference_threshold=0.3,
            electronegativity_difference_threshold=1.0,
        )
        super().__init__(structure, symprec=0.01, angle_tolerance=5.0)
        super().__init__(structure, symprec=0.01, angle_tolerance=5.0)
