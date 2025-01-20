# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import gzip
import os
import pickle
import shutil
import weakref
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, DefaultDict, Iterator, Mapping

import lmdb  # type: ignore [import]
from monty.json import MontyDecoder
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedStructureEntry
from tqdm.autonotebook import tqdm

from mattergen.evaluation.reference.reference_dataset import ReferenceDataset, ReferenceDatasetImpl
from mattergen.evaluation.utils.lmdb_utils import lmdb_get, lmdb_open, lmdb_read_metadata


def gzip_compress(file_path: str | os.PathLike, output_dir: str | os.PathLike) -> Path:
    """Compresses a file using gzip. Returns the compressed file path."""
    output_path = Path(output_dir) / (Path(file_path).name + ".gz")
    with open(file_path, "rb") as fin:
        with gzip.open(output_path, "wb") as fout:
            fout.write(fin.read())
    return output_path


def gzip_decompress(gzip_file_path: str | os.PathLike, output_dir: str | os.PathLike) -> Path:
    """Decompresses a gzipped file. Returns the decompressed file path."""
    output_path = Path(output_dir) / Path(gzip_file_path).name[:-3]  # remove .gz
    with gzip.open(gzip_file_path, "rb") as fin:
        with open(output_path, "wb") as fout:
            fout.write(fin.read())
    return output_path


class LmdbNotFoundError(Exception):
    pass


def lmdb_open(db_path: str | os.PathLike, readonly: bool = False) -> lmdb.Environment:
    if readonly:
        return lmdb.open(
            str(db_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
    else:
        return lmdb.open(
            str(db_path),
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
        )


def lmdb_read_metadata(db_path: str | os.PathLike, key: str, default=None) -> Any:
    with lmdb_open(db_path, readonly=True) as db:
        with db.begin() as txn:
            result = lmdb_get(txn, key, default=default)
    return result


def lmdb_get(
    txn: lmdb.Transaction, key: str, default: Any = None, raise_if_missing: bool = True
) -> Any:
    """
    Fetches a record from a database.

    Args:
        txn: LMDB transaction (use env.begin())
        key: key of the data to be fetched.
        default: default value to be used if the record doesn't exist.
        raise_if_missing: raise LmdbNotFoundError if the record doesn't exist
            and no default value was given.

    Returns:
        the value of the retrieved data.
    """
    value = txn.get(key.encode("ascii"))
    if value is None:
        if default is None and raise_if_missing:
            raise LmdbNotFoundError(
                f"Key {key} not found in database but default was not provided."
            )
        return default
    return pickle.loads(value)


def lmdb_put(txn: lmdb.Transaction, key: str, value: Any) -> bool:
    """
    Stores a record in a database.

    Args:
        txn: LMDB transaction (use env.begin())
        key: key of the data to be stored.
        value: value of the data to be stored (needs to be picklable).

    Returns:
        True if it was written.
    """
    return txn.put(
        key.encode("ascii"),
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
    )


class LMDBGZSerializer():
    def __init__(
        self,
    ):
        pass

    def serialize(self, ref_dataset: ReferenceDataset, dataset_path: str | os.PathLike) -> None:
        """Writes a dataset to a file using the gzip-compressed LMDB format."""
        lmdb_file_path = str(dataset_path)[:-3]  # remove .gz
        with lmdb_open(lmdb_file_path, readonly=False) as env:
            # Store the metadata
            with env.begin(write=True) as txn:
                lmdb_put(txn, "name", ref_dataset.name)

            # Entries are stored under the key "{chemsys}.{reduced_formula}.{n}"
            # where n is the index of the entry within the reduced_formula.
            # This enables us to retrieve entries for a given reduced_formula
            # or chemical system.
            counter: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
            for entry in tqdm(ref_dataset, desc="Serializing dataset", total=len(ref_dataset)):
                entry.structure.unset_charge()
                structure_without_oxidation_states = entry.structure.remove_oxidation_states()
                entry = ComputedStructureEntry.from_dict(
                    {
                        **entry.as_dict(),
                        "structure": structure_without_oxidation_states,
                        "composition": structure_without_oxidation_states.composition,
                    }
                )
                chemsys = "-".join(sorted({el.symbol for el in entry.composition.elements}))
                reduced_formula = entry.composition.reduced_formula
                n = counter[chemsys][reduced_formula]
                key = f"{chemsys}.{reduced_formula}.{n}"
                with env.begin(write=True) as txn:
                    lmdb_put(txn, key, entry.as_dict())
                counter[chemsys][reduced_formula] += 1
            # Store the list of chemical systems
            chemical_systems = list(counter.keys())
            with env.begin(write=True) as txn:
                lmdb_put(txn, "chemical_systems", chemical_systems)
            for chemsys, length_by_reduced_formula in tqdm(
                counter.items(), desc="Saving indexes", total=len(counter)
            ):
                # Store the list of reduced formulas in this chemical system
                reduced_formulas = list(length_by_reduced_formula.keys())
                with env.begin(write=True) as txn:
                    lmdb_put(txn, f"{chemsys}.reduced_formulas", reduced_formulas)
                # Store the number of entries for each reduced formula
                for reduced_formula, length in length_by_reduced_formula.items():
                    with env.begin(write=True) as txn:
                        lmdb_put(txn, f"{chemsys}.{reduced_formula}.length", length)
        gzip_compress(lmdb_file_path, Path(dataset_path).parent)

    def deserialize(self, dataset_path: str | os.PathLike) -> ReferenceDataset:
        """Reads a dataset from a file using the gzip-compressed LMDB format."""
        tempdir = mkdtemp()
        lmdb_path = gzip_decompress(dataset_path, tempdir)
        name = lmdb_read_metadata(lmdb_path, "name")
        return ReferenceDataset(
            name=name,
            impl=LMDBBackedReferenceDatasetImpl(lmdb_path, cleanup_dir=True),
        )


class LMDBBackedReferenceDatasetImpl(ReferenceDatasetImpl):
    """Implementation of ReferenceDataset backed by LMDB.

    Expected LMDB structure:
        {
            "chemical_systems": ["Li-P", "Li-S", ...],
            "Li-P.reduced_formulas": ["LiP", "LiP2", ...],
            "Li-P.LiP.length": 4,
            "Li-P.LiP.0": "<pickled dictionary representation of a ComputedStructureEntry>",
            ...
            "Li-P.LiP.3": "<pickled dictionary representation of a ComputedStructureEntry>",
            "Li-P.LiP2.length": 1,
            ...
            "Li-S.Li2S.length": 2,
            ...
        }
    """

    def __init__(self, lmdb_path: Path, cleanup_dir: bool = False):
        """Initializes the LMDB-backed reference dataset.

        Args:
            lmdb_path: path to the LMDB database.
            cleanup_dir: whether to delete the directory containing the database when this object
                is garbage collected (default: False).
        """
        self.env = lmdb_open(lmdb_path, readonly=True)
        self.num_entries_by_chemsys_reduced_formulas = (
            self._build_num_entries_by_chemsys_reduced_formulas(lmdb_path)
        )
        self.total_num_entries = sum(
            sum(d.values()) for d in self.num_entries_by_chemsys_reduced_formulas.values()
        )
        # close the LMDB environment when this object is garbage collected
        weakref.finalize(self, self._cleanup, self.env, cleanup_dir)

    def _build_num_entries_by_chemsys_reduced_formulas(
        self, lmdb_path: Path
    ) -> dict[str, dict[str, int]]:
        chemical_systems = lmdb_read_metadata(lmdb_path, "chemical_systems")
        result: defaultdict[str, dict[str, int]] = defaultdict(dict)
        with self.env.begin() as txn:
            for chemsys in chemical_systems:
                reduced_formulas = lmdb_read_metadata(lmdb_path, f"{chemsys}.reduced_formulas")
                for reduced_formula in reduced_formulas:
                    result[chemsys][reduced_formula] = lmdb_get(
                        txn, f"{chemsys}.{reduced_formula}.length"
                    )
        # convert to an ordinary dictionary
        return {key: val for key, val in result.items()}

    def __iter__(self) -> Iterator[ComputedStructureEntry]:
        """Iterates over the entries in the dataset."""
        for (
            chemsys,
            num_entries_by_reduced_formula,
        ) in self.num_entries_by_chemsys_reduced_formulas.items():
            for reduced_formula in num_entries_by_reduced_formula:
                yield from self.get_entries_by_chemsys_reduced_formula(chemsys, reduced_formula)

    def __len__(self) -> int:
        return self.total_num_entries

    @property
    def chemical_systems(self) -> tuple[str, ...]:
        return tuple(self.num_entries_by_chemsys_reduced_formulas.keys())

    @cached_property
    def reduced_formulas(self) -> tuple[str, ...]:
        return tuple(
            [
                reduced_formula
                for num_entries_by_reduced_formula in self.num_entries_by_chemsys_reduced_formulas.values()
                for reduced_formula in num_entries_by_reduced_formula
            ]
        )

    def get_entries_by_chemsys(self, chemsys: str) -> Iterator[ComputedStructureEntry]:
        for reduced_formula in self.num_entries_by_chemsys_reduced_formulas[chemsys].keys():
            yield from self.get_entries_by_chemsys_reduced_formula(chemsys, reduced_formula)

    def get_entries_by_reduced_formula(
        self, reduced_formula: str
    ) -> Iterator[ComputedStructureEntry]:
        chemsys = Composition(reduced_formula).chemical_system
        yield from self.get_entries_by_chemsys_reduced_formula(chemsys, reduced_formula)

    def get_entries_by_chemsys_reduced_formula(
        self, chemsys: str, reduced_formula: str
    ) -> Iterator[ComputedStructureEntry]:
        length = self.num_entries_by_chemsys_reduced_formulas[chemsys][reduced_formula]
        for i in range(length):
            with self.env.begin() as txn:
                entry_dict = lmdb_get(txn, f"{chemsys}.{reduced_formula}.{i}")
            yield MontyDecoder().process_decoded(entry_dict)

    @cached_property
    def entries_by_reduced_formula(self) -> "LMDBBackedReducedFormulaLookup":
        """Returns a mapping from reduced formula to entries."""
        return LMDBBackedReducedFormulaLookup(self)

    @cached_property
    def entries_by_chemsys(self) -> "LMDBBackedChemicalSystemLookup":
        """Returns a mapping from chemical system to entries."""
        return LMDBBackedChemicalSystemLookup(self)

    @classmethod
    def _cleanup(cls, env: lmdb.Environment, cleanup_dir: bool) -> None:
        """Closes the LMDB environment and deletes the directory containing the database.

        This needs to be a class method to prevent additional reference to the object.
        """
        try:
            database_dir = Path(env.path()).parent
        except lmdb.Error:
            # The environment has already been closed.
            return
        print(f"Closing LMDB environment {env.path()}")
        env.close()
        if cleanup_dir:
            shutil.rmtree(database_dir)

    def cleanup(self, cleanup_dir: bool = False) -> None:
        """Closes the LMDB environment and optionally cleanup the directory containing the database."""
        self._cleanup(self.env, cleanup_dir)


class WeakRefImplMixin:
    """A mixin class that makes the reference to the underlying
    LMDBBackedReferenceDatasetImpl object weak."""

    def __init__(self, impl: LMDBBackedReferenceDatasetImpl):
        # We need to use a weak reference to avoid cyclic reference that
        # prevents LMDBBackedReferenceDatasetImpl from being garbage collected.
        self._impl = weakref.ref(impl)

    @property
    def impl(self) -> LMDBBackedReferenceDatasetImpl:
        # Returns the LMDBBackedReferenceDatasetImpl object ensuring that
        # the reference is still valid.
        impl = self._impl()
        assert impl is not None
        return impl
    

class LMDBBackedChemicalSystemLookup(WeakRefImplMixin, Mapping[str, list[ComputedStructureEntry]]):
    """A lazy immutable mapping from chemical system to entries. It is
    lazy in the sense that the entries are read from the disk only when
    the user requests them."""

    def __init__(self, impl: LMDBBackedReferenceDatasetImpl):
        super().__init__(impl)
        self.chemical_systems = frozenset(self.impl.chemical_systems)

    def __len__(self) -> int:
        return len(self.impl.chemical_systems)

    def __iter__(self) -> Iterator[str]:
        # keep the original order
        return iter(self.impl.chemical_systems)

    def __contains__(self, chemical_system: object) -> bool:
        return chemical_system in self.chemical_systems

    def __getitem__(self, chemical_system: str) -> list[ComputedStructureEntry]:
        return list(self.impl.get_entries_by_chemsys(chemical_system))


class LMDBBackedReducedFormulaLookup(WeakRefImplMixin, Mapping[str, list[ComputedStructureEntry]]):
    """A lazy immutable mapping from reduced formula to entries. It is
    lazy in the sense that the entries are read from the disk only when
    the user requests them."""

    def __init__(self, impl: LMDBBackedReferenceDatasetImpl):
        super().__init__(impl)
        self.reduced_formulas = frozenset(self.impl.reduced_formulas)

    def __len__(self) -> int:
        return len(self.reduced_formulas)

    def __iter__(self) -> Iterator[str]:
        # keep the original order
        return iter(self.impl.reduced_formulas)

    def __contains__(self, reduced_formula: object) -> bool:
        return reduced_formula in self.reduced_formulas

    def __getitem__(self, reduced_formula: str) -> list[ComputedStructureEntry]:
        """Returns a list of entries with the given reduced formula."""
        return list(self.impl.get_entries_by_reduced_formula(reduced_formula))
