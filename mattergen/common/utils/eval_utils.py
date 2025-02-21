# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence
from zipfile import ZipFile

import ase.io
import hydra
import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from mattergen.common.globals import (
    GENERATED_CRYSTALS_EXTXYZ_FILE_NAME,
    GENERATED_CRYSTALS_ZIP_FILE_NAME,
)
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.globals import get_device
from mattergen.diffusion.lightning_module import DiffusionLightningModule

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_structure(
    lengths: torch.Tensor,
    angles: torch.Tensor,
    atom_types: torch.Tensor,
    frac_coords: torch.Tensor,
) -> Structure:
    return Structure(
        lattice=Lattice.from_parameters(
            **{a: v for a, v in zip(["a", "b", "c"], lengths)},
            **{a: v for a, v in zip(["alpha", "beta", "gamma"], angles)},
        ),
        species=atom_types,
        coords=frac_coords,
        coords_are_cartesian=False,
    )


def load_model_diffusion(
    args: MatterGenCheckpointInfo,
) -> DiffusionLightningModule:
    assert args.load_epoch is not None
    ckpt = args.checkpoint_path
    logger.info(f"Loading model from checkpoint: {ckpt}")
    cfg = args.config
    try:
        model, incompatible_keys = DiffusionLightningModule.load_from_checkpoint_and_config(
            ckpt,
            map_location=get_device(),
            config=cfg.lightning_module,
            strict=args.strict_checkpoint_loading,
        )
    except hydra.errors.HydraException as e:
        raise
    if len(incompatible_keys.unexpected_keys) > 0:
        raise ValueError(f"Unexpected keys in checkpoint: {incompatible_keys.unexpected_keys}.")
    if len(incompatible_keys.missing_keys) > 0:
        raise ValueError(f"Missing keys in checkpoint: {incompatible_keys.missing_keys}.")

    return model


def get_crystals_list(
    frac_coords, atom_types, lengths, angles, num_atoms
) -> list[dict[str, np.ndarray]]:
    """
    args:
        frac_coords: (num_atoms, 3)
        atom_types: (num_atoms)
        lengths: (num_crystals)
        angles: (num_crystals)
        num_atoms: (num_crystals)
    """
    assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
    assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

    start_idx = 0
    crystal_array_list = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
        cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]

        crystal_array_list.append(
            {
                "frac_coords": cur_frac_coords.detach().cpu().numpy(),
                "atom_types": cur_atom_types.detach().cpu().numpy(),
                "lengths": cur_lengths.detach().cpu().numpy(),
                "angles": cur_angles.detach().cpu().numpy(),
            }
        )
        start_idx = start_idx + num_atom
    return crystal_array_list


def save_structures(output_path: Path, structures: Sequence[Structure]) -> None:
    """Save structures to disk in a extxyz file and a compressed zip file containing cif files.

    Args:
        output_path: path to a directory where the results are written.
        structures: sequence of structures.
    """
    ase_atoms = [AseAtomsAdaptor.get_atoms(x) for x in structures]
    try:
        ase.io.write(output_path / GENERATED_CRYSTALS_EXTXYZ_FILE_NAME, ase_atoms)

        with ZipFile(output_path / GENERATED_CRYSTALS_ZIP_FILE_NAME, "w") as zip_obj:
            for ix, ase_atom in enumerate(ase_atoms):
                ase.io.write(f"/tmp/gen_{ix}.cif", ase_atom, format="cif")
                zip_obj.write(f"/tmp/gen_{ix}.cif", arcname=f"gen_{ix}.cif")
    except IOError as e:
        print(f"Got error {e} writing the generated structures to disk.")


def load_structures(input_path: Path) -> Sequence[Structure]:
    """Load structures from disk.

    Args:
        output_path: path to a file or directory where the results are written.

    Returns:
        sequence of structures.
    """
    # if the path is an xyz or extxyz file, read it directly
    if input_path.suffix == ".xyz" or input_path.suffix == ".extxyz":
        ase_atoms = ase.io.read(input_path, ":")
        return [AseAtomsAdaptor.get_structure(x) for x in ase_atoms]

    # if the path is a zipped folder, extract it into a temporary directory
    elif input_path.suffix == ".zip":
        with TemporaryDirectory() as tmpdirname:
            with ZipFile(input_path, "r") as zip_obj:
                zip_obj.extractall(tmpdirname)
            return extract_structures_from_folder(tmpdirname)

    # if the path is a directory, read all files in it
    elif input_path.is_dir():
        return extract_structures_from_folder(input_path)

    else:
        raise ValueError(f"Invalid input path {input_path}")


def extract_structures_from_folder(dirname: str) -> Sequence[Structure]:
    structures = []
    for filename in os.listdir(dirname):
        if filename.endswith(".cif"):
            try:
                structures.append(Structure.from_file(f"{dirname}/{filename}"))
            except ValueError as e:
                logger.warning(f"Failed to read {filename} as a CIF file: {e}")
        elif filename.endswith(".extxyz") or filename.endswith(".xyz"):
            ase_atoms = ase.io.read(
                f"{dirname}/{filename}", 0
            )  #  We assume that the file contains only one structure
            structures.append(AseAtomsAdaptor.get_structure(ase_atoms))
    return structures
