from pathlib import Path

from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry

from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer


def test_deserialize_does_not_reopen_same_lmdb_while_environment_is_active(
    tmp_path: Path,
) -> None:
    serializer = LMDBGZSerializer()
    dataset_path = tmp_path / "reference.lmdb.gz"
    entry = ComputedStructureEntry(
        structure=Structure(
            lattice=Lattice.cubic(3.5),
            species=["Fe", "O"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        ),
        energy=0.0,
    )
    serializer.serialize(
        ReferenceDataset.from_entries("reference", [entry]),
        dataset_path,
    )

    reference = serializer.deserialize(dataset_path)
    assert reference.name == "reference"
    assert reference.impl.chemical_systems == ("Fe-O",)
    assert len(reference) == 1
    reference.impl.cleanup(cleanup_dir=True)
