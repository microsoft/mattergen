import numpy as np
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.kpath import KPathSeek
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read
from mattersim.forcefield.potential import Potential
from mattersim.datasets.utils.build import build_dataloader
from tqdm import tqdm
from mattergen.evaluation.utils.relaxation import relax_structures
from mattergen.common.utils.globals import get_device
import copy
import torch
import csv
import os
from datetime import datetime
from pathlib import Path


def _phonopy_atoms_to_ase(phonopy_atoms):
    return Atoms(
        symbols=phonopy_atoms.symbols,
        positions=phonopy_atoms.positions,
        cell=phonopy_atoms.cell,
        pbc=True,
    )


def _to_phonopy_atoms(structure):
    """Convert pymatgen Structure to PhonopyAtoms"""
    return PhonopyAtoms(
        symbols=[str(s) for s in structure.species],
        cell=structure.lattice.matrix,
        scaled_positions=structure.frac_coords,
    )


def _get_mattersim_potential(potential_load_path, device):
    return Potential.from_checkpoint(
        device=device, load_path=potential_load_path, load_training_state=False
    )


def _calculate_forces_mattersim(displaced_supercells, potential):
    atoms_list = [_phonopy_atoms_to_ase(scell) for scell in displaced_supercells]
    dataloader = build_dataloader(atoms_list, batch_size=16, only_inference=True)

    energy_batch, forces_batch, stress_batch = potential.predict_properties(
        dataloader, include_forces=True, include_stresses=False
    )
    return forces_batch


def _clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _setup_phonon_calculation(structure, supercell_matrix, distance):
    """Setup phonopy calculation with displacements"""
    phonopy_structure = _to_phonopy_atoms(structure)
    phonon = Phonopy(phonopy_structure, supercell_matrix=supercell_matrix)
    phonon.generate_displacements(distance=distance)
    return phonon


def _get_standard_band_path(structure):
    kpath = KPathSeek(structure, symprec=0.01)._kpath

    path = copy.deepcopy(kpath["path"])

    for set_idx, label_set in enumerate(kpath["path"]):
        for lbl_idx, label in enumerate(label_set):
            path[set_idx][lbl_idx] = kpath["kpoints"][label]

    return path, [l for sublist in kpath["path"] for l in sublist]


def _run_band_structure_and_dos(phonon: Phonopy, structure, mesh, npoints=51):
    bands, labels = _get_standard_band_path(structure)

    # Generate qpoints and connections
    qpoints, connections = get_band_qpoints_and_path_connections(bands, npoints=npoints)
    phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)

    # DOS calculation
    phonon.run_mesh(mesh)
    phonon.run_total_dos()


def _get_min_frequency(phonon):
    """Get minimum frequency from phonon spectrum"""
    mesh_dict = phonon.get_mesh_dict()
    frequencies = mesh_dict["frequencies"]
    min_freq = np.min(frequencies)
    return min_freq


def _initialize_frequency_log(filename="phonon_frequencies.csv"):
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["timestamp", "formula", "min_frequency_THz"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    print(f"📋 Initialized frequency log: {filename}")


def _save_frequency_to_file(
    structure_formula, min_frequency, filename="phonon_frequencies.csv", append=True
):
    file_exists = os.path.exists(filename)
    mode = "a" if (append and file_exists) else "w"

    with open(filename, mode, newline="") as csvfile:
        fieldnames = ["timestamp", "formula", "min_frequency_THz"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if new file or overwriting
        if not file_exists or not append:
            writer.writeheader()

        # Write data
        writer.writerow(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "formula": structure_formula,
                "min_frequency_THz": f"{min_frequency:.6f}",
            }
        )


def _calculate_phonon_spectrum_with_potential(
    structure,
    potential,
    supercell_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
    mesh=[20, 20, 20],
    distance=0.02,
):
    # Setup phonopy calculation
    phonon = _setup_phonon_calculation(structure, supercell_matrix, distance)

    # Calculate forces for displaced structures using pre-loaded MatterSim
    forces_list = _calculate_forces_mattersim(phonon.supercells_with_displacements, potential)
    _clear_gpu_memory()

    # Set forces and produce force constants
    phonon.forces = forces_list
    phonon.produce_force_constants()

    # Run band structure and DOS calculations
    _run_band_structure_and_dos(phonon, structure, mesh)

    # Get minimum frequency
    min_freq = _get_min_frequency(phonon)

    return phonon, min_freq


def stability_screening(
    structures,
    potential_load_path,
    device=get_device(),
    output_file="phonon_frequencies.csv",
    relax_structure=False,
    relax_fmax=1e-5,
    **phonopy_kwargs,
):
    _initialize_frequency_log(output_file)

    print(f"Loading MatterSim potential from {potential_load_path}...")
    potential = _get_mattersim_potential(potential_load_path=potential_load_path, device=device)
    print("MatterSim potential loaded successfully!")

    # Phase 1: Relax all structures if requested
    if relax_structure:
        structures, _ = relax_structures(
            structures=structures,
            potential_load_path=potential_load_path,
            fmax=relax_fmax,
        )

    results = []

    # Phase 2: Calculate phonons for all relaxed structures
    print(f"\n🔊 Phase 2: Calculating phonons for {len(structures)} structures...")
    for structure in tqdm(structures):
        min_freq = None

        try:
            _, min_freq = _calculate_phonon_spectrum_with_potential(
                structure, potential, **phonopy_kwargs
            )

            results.append(
                {
                    "formula": structure.composition.reduced_formula,
                    "min_frequency": min_freq,
                }
            )

        except Exception as e:
            print(f"  ⚠ Error calculating phonons: {e}")

            results.append(
                {
                    "formula": structure.composition.reduced_formula,
                    "error": str(e),
                }
            )

        _save_frequency_to_file(
            structure_formula=structure.composition.reduced_formula,
            min_frequency=min_freq or float("nan"),  # Use NaN for errors
            filename=output_file,
            append=True,
        )

        _clear_gpu_memory()

    return results


def example_usage():
    from pymatgen.core import Lattice

    lattice_diamond = Lattice.cubic(a=3.567)
    structure = Structure(
        lattice=lattice_diamond, species=["C", "C"], coords=[[0, 0, 0], [0.25, 0.25, 0.25]]
    )

    potential_load_path = "MatterSim-v1.0.0-5M.pth"
    device = get_device()
    relax_structure = True
    relax_fmax = 0.005

    potential = _get_mattersim_potential(
        potential_load_path=potential_load_path,
        device=device,
    )

    # Optionally relax structure first
    if relax_structure:
        relaxed_structures, _ = relax_structures(
            structures=[structure],
            potential_load_path=potential_load_path,
            fmax=relax_fmax,
        )
        structure = relaxed_structures[0]

    phonon, min_freq = _calculate_phonon_spectrum_with_potential(
        structure,
        potential,
    )

    print(f"Structure minimum frequency: {min_freq:.3f} THz")

    phonon.plot_band_structure().savefig("phonon_spectrum.png", dpi=300)


if __name__ == "__main__":
    # example_usage()

    path = Path("new_work/generated_structures/topological_dgf2/candidates_highly_relaxed.extxyz")
    ase_atoms_list = read(path, index=":")
    structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in ase_atoms_list]

    output_filename = f"new_work/generated_structures/topological_dgf2/phonon_stability_results.csv"

    results = stability_screening(
        structures,
        potential_load_path="MatterSim-v1.0.0-5M.pth",
        output_file=output_filename,
        relax_structure=False,  # recommended to relax structures separately and load then load from file (use the same potential)
    )

    for result in results[:50]:  # Print first 50 results
        if "error" in result:
            print(f"❌ {result['formula']}: Error - {result['error']}")
        else:
            print(f"📊 {result['formula']}: min freq: {result['min_frequency']:.3f} THz")
