from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pytest
from pymatgen.core import Lattice, Structure

from mattergen.common.globals import (
    GENERATED_CRYSTALS_EXTXYZ_FILE_NAME,
    GENERATED_CRYSTALS_ZIP_FILE_NAME,
)
from mattergen.common.utils.eval_utils import save_structures


def test_save_structures_writes_cifs_directly_to_archive(tmp_path: Path) -> None:
    structures = [
        Structure(Lattice.cubic(3), ["Li"], [[0, 0, 0]]),
        Structure(Lattice.cubic(4), ["O"], [[0, 0, 0]]),
    ]

    save_structures(tmp_path, structures)

    assert (tmp_path / GENERATED_CRYSTALS_EXTXYZ_FILE_NAME).is_file()
    with ZipFile(tmp_path / GENERATED_CRYSTALS_ZIP_FILE_NAME) as archive:
        assert archive.namelist() == ["gen_0.cif", "gen_1.cif"]
        assert all(archive.read(name) for name in archive.namelist())


def test_save_structures_propagates_write_errors(tmp_path: Path) -> None:
    structures = [Structure(Lattice.cubic(3), ["Li"], [[0, 0, 0]])]

    with (
        patch(
            "mattergen.common.utils.eval_utils.ase.io.write",
            side_effect=OSError("disk full"),
        ),
        pytest.raises(OSError, match="disk full"),
    ):
        save_structures(tmp_path, structures)
