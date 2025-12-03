# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pymatgen.core.structure import Structure

from mattergen.common.utils.globals import get_device
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
    structure_matcher: (
        OrderedStructureMatcher | DisorderedStructureMatcher
    ) = DefaultDisorderedStructureMatcher(),
    save_as: str | None = None,
    potential_load_path: str | None = None,
    device: str = str(get_device()),
    structures_output_path: str | None = None,
    **kwargs,
) -> dict[str, float | int]:
    """Evaluate the structures against a reference dataset.

    Args:
        structures: List of structures to evaluate.
        relax: Whether to relax the structures before evaluation. Note that if this is run, `energies` will be ignored.
        energies: Energies of the structures if already relaxed and computed externally (e.g., from DFT).
        reference: Reference dataset. If this is None, the default reference dataset will be used.
        structure_matcher: Structure matcher to use for matching the structures.
        save_as: Save the metrics as a JSON file.
        potential_load_path: Path to the Machine Learning potential to use for relaxation.
        device: Device to use for relaxation.
        structures_output_path: Path to save the relaxed structures.

    Returns:
        metrics: a dictionary of metrics and their values.
    """
    if relax and energies is not None:
        raise ValueError("Cannot accept energies if relax is True.")
    if relax:
        relaxed_structures, energies = relax_structures(
            structures,
            device=device,
            potential_load_path=potential_load_path,
            output_path=structures_output_path,
            **kwargs,
        )
    else:
        relaxed_structures = structures
    evaluator = MetricsEvaluator.from_structures_and_energies(
        structures=relaxed_structures,
        energies=energies,
        original_structures=structures,
        reference=reference,
        structure_matcher=structure_matcher,
    )
    return evaluator.as_dataframe(metrics=evaluator.available_metrics)
