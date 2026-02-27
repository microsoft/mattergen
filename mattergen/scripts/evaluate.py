# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from pathlib import Path
from typing import Literal

import fire
import numpy as np
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

from mattergen.common.utils.eval_utils import load_structures
from mattergen.common.utils.globals import get_device
from mattergen.evaluation.evaluate import evaluate
from mattergen.evaluation.reference.correction_schemes import TRI110Compatibility2024
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DefaultOrderedStructureMatcher,
)


def main(
    structures_path: str,
    relax: bool = True,
    energies_path: str | None = None,
    structure_matcher: Literal["ordered", "disordered"] = "disordered",
    save_as: str | None = None,
    save_detailed_as: str | None = None,
    potential_load_path: (
        Literal["MatterSim-v1.0.0-1M.pth", "MatterSim-v1.0.0-5M.pth"] | None
    ) = None,
    reference_dataset_path: str | None = None,
    device: str = str(get_device()),
    structures_output_path: str | None = None,
    energy_correction_scheme: Literal["MP2020", "TRI2024"] = "MP2020",
):
    structures = load_structures(Path(structures_path))
    energies = np.load(energies_path) if energies_path else None
    structure_matcher = (
        DefaultDisorderedStructureMatcher()
        if structure_matcher == "disordered"
        else DefaultOrderedStructureMatcher()
    )
    reference = None
    if reference_dataset_path:
        reference = LMDBGZSerializer().deserialize(reference_dataset_path)
    
    match energy_correction_scheme:
        case "MP2020":
            energy_correction_scheme = MaterialsProject2020Compatibility()
        case "TRI2024":
            energy_correction_scheme = TRI110Compatibility2024()
    
    metrics = evaluate(
        structures=structures,
        relax=relax,
        energies=energies,
        structure_matcher=structure_matcher,
        save_as=save_as,
        save_detailed_as=save_detailed_as,
        potential_load_path=potential_load_path,
        reference=reference,
        device=device,
        structures_output_path=structures_output_path,
        energy_correction_scheme=energy_correction_scheme,
    )
    print(json.dumps(metrics, indent=2))


def _main():
    fire.Fire(main)


if __name__ == "__main__":
    _main()
