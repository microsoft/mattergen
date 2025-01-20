# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pymatgen.core.structure import Structure

from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.utils.relaxation import relax_structures
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DisorderedStructureMatcher,
    OrderedStructureMatcher,
)


def evaluate(
    structures: list[Structure],
    relax: bool = True,
    energies: list[float] | None = None,
    reference: ReferenceDataset | None = None,
    structure_matcher: OrderedStructureMatcher
        | DisorderedStructureMatcher = DefaultDisorderedStructureMatcher(),
    save_as: str | None = None,
    potential_load_path: str | None = None,
    device: str = "cuda",
) -> dict[str, float | int]:
    """Evaluate the structures against a reference dataset.

    Args:
        structures: List of structures to evaluate.
        relax: Whether to relax the structures before evaluation. Note that if this is run, `energies` will be ignored.
        energies: Energies of the structures if already relaxed and computed externally (e.g., from DFT).
        reference_dataset: Reference dataset.
        ordered_structure_matcher: Matcher for ordered structures.
        disordered_structure_matcher: Matcher for disordered structures.
        n_jobs: Number of parallel jobs.

    Returns:
        metrics: a dictionary of metrics and their values.
    """
    if relax and energies is not None:
        raise ValueError("Cannot accept energies if relax is True.")
    if relax:
        relaxed_structures, energies = relax_structures(structures, device=device, load_path=potential_load_path)
    else:
        relaxed_structures = structures
    evaluator = MetricsEvaluator.from_structures_and_energies(
        structures=relaxed_structures, 
        energies=energies,
        original_structures=structures,
        reference=reference,
        structure_matcher=structure_matcher
    )
    return evaluator.compute_metrics(
        metrics = evaluator.available_metrics,
        save_as = save_as,
        pretty_print = True,
    )
