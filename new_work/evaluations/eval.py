from mattergen.common.utils.globals import get_device
from mattergen.evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher
from mattergen.evaluation.evaluate import evaluate
from mattergen.evaluation.utils.relaxation import relax_structures
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
from pathlib import Path
from tqdm import tqdm


def relax(
    folder_path, fmax=1e-5, device=get_device(), potential_load_path="MatterSim-v1.0.0-5M.pth"
):
    """Sample snippet for relaxing structures to a higher accuracy for phonon calculations."""
    structures_path = folder_path / "candidates.extxyz"

    structures = [
        AseAtomsAdaptor.get_structure(atoms) for atoms in tqdm(read(structures_path, index=":"))
    ]

    structures_output_path = folder_path / "candidates_highly_relaxed.extxyz"

    relax_structures(
        structures,
        device=device,
        potential_load_path=potential_load_path,
        output_path=structures_output_path,
        fmax=fmax,
    )


def eval(folder_path, fmax):
    structures_path = folder_path / "generated_crystals.extxyz"

    structures = [
        AseAtomsAdaptor.get_structure(atoms) for atoms in tqdm(read(structures_path, index=":"))
    ]

    structure_matcher = DefaultDisorderedStructureMatcher()
    device = str(get_device())
    print(f"Using device: {device}")

    df = evaluate(
        structures=structures,
        relax=True,
        structure_matcher=structure_matcher,
        device=device,
        potential_load_path="MatterSim-v1.0.0-5M.pth",
        structures_output_path=folder_path / "relaxed_crystals.extxyz",
        fmax=fmax,
    )

    df = df.drop(columns=["entry"], errors="ignore")  # contains unused information
    df.to_csv(folder_path / "metrics.csv", index=False)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", message="Failed to guess oxidation states*")

    folder_path = Path("new_work/generated_structures/topological_dgf2")

    eval(folder_path, fmax=0.05)
