from ase.io import read
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def create_space_group_info(structures: list[Structure], output_dir: Path):
    space_groups_01 = [
        s.get_space_group_info(symprec=0.1)
        for s in tqdm(structures, desc="Calculating space groups with symprec=0.1")
    ]
    space_groups_001 = [
        s.get_space_group_info(symprec=0.01)
        for s in tqdm(structures, desc="Calculating space groups with symprec=0.01")
    ]

    space_group_symbols_01 = [sg[0] for sg in space_groups_01]
    space_group_numbers_01 = [sg[1] for sg in space_groups_01]
    space_group_symbols_001 = [sg[0] for sg in space_groups_001]
    space_group_numbers_001 = [sg[1] for sg in space_groups_001]

    df = pd.DataFrame(
        {
            "formula": [s.composition.reduced_formula for s in structures],
            "space_group_symbol_01": space_group_symbols_01,
            "space_group_number_01": space_group_numbers_01,
            "space_group_symbol_001": space_group_symbols_001,
            "space_group_number_001": space_group_numbers_001,
        }
    )

    df.to_csv(output_dir / "space_group_info.csv", index=False)


def create_space_group_info_for_dir(base_dir: Path):
    path = base_dir / "relaxed_structures.extxyz"
    structures = read(path, index=":", format="extxyz")
    structures = [AseAtomsAdaptor.get_structure(s) for s in structures]
    output_dir = base_dir

    create_space_group_info(structures, output_dir)
