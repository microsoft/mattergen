# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import bisect
import os
import pickle
from abc import abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Generic, TypeVar

import lmdb  # type: ignore [import]
from tqdm import tqdm  # type: ignore [import]

T = TypeVar("T")
DataPoint = TypeVar("DataPoint")


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


class LmdbNotFoundError(Exception):
    pass


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


def get_length(env: lmdb.Environment) -> int:
    """
    Returns the value of the special record "length".

    Args:
        env: LMDB environment (use lmdb.open())

    Returns:
        the value of the "length" record or zero if the record does not exist.
    """
    with env.begin() as txn:
        return lmdb_get(txn, "length", default=0)


def list_db_paths(data_dir: str | os.PathLike) -> list[Path]:
    return sorted(Path(data_dir).glob("*.lmdb"))


def get_envs(data_dir: str | os.PathLike) -> Iterator[lmdb.Environment]:
    """
    Creates LMDB environments stored in a directory.

    Args:
        data_dir: directory where the .lmdb files are stored.

    Returns:
        an iterator over LMDB environments.
    """
    for lmdb_path in list_db_paths(data_dir):
        yield lmdb_open(lmdb_path, readonly=True)


def get_indices(cum_lengths: Sequence[int], index: int) -> tuple[int, int]:
    """
    Given a sequence of cumulative sequence lengths and a linear index over a sequence
    of variable length databases, returns a pair of (db_index, el_index).
    """
    db_index = bisect.bisect(cum_lengths, index)
    el_index = index - cum_lengths[db_index - 1] if db_index > 0 else index
    return (db_index, el_index)


class Metadata(Generic[DataPoint]):
    @abstractmethod
    def __init__(self, value=None):
        """Initialize with an optional value"""

    @classmethod
    def from_value(cls, value):
        return cls(value)

    @property
    @abstractmethod
    def name(self):
        """The name of the metadata"""

    @property
    def is_frozen(self):
        return False

    @property
    def value(self):
        """The value of the metadata"""
        return self._value

    def check(self, num_points: int):
        """Check consistency of the metadata"""
        pass

    def update(self, index: int, sample: DataPoint):
        """Update the metadata with the new datapoint"""
        pass


# From https://github.com/Open-Catalyst-Project/ocp/blob/master/scripts/preprocess_ef.py
def write_data_points_to_lmdb(
    db_path: str,
    samples: Iterable[DataPoint],
    pid: int | None = None,
    metadata: list[Metadata] | None = None,
) -> int:
    """
    Creates or appends to a database of data points keyed by the string representation of linear
    index over the data points.

    Args:
        start_index: start index for this group of samples. Should match the length of the existing database.
        db_path: path to store the database.
        samples: iterable over the data points to be stored.
        metadata: (optional) list of metadata objects implementing
            `check` and `update` methods.

    Returns:
        the number of samples stored in the database.
    """
    with lmdb_open(db_path) as db:
        start_index = get_length(db)

        metadata = check_and_init_metadata(db, metadata, start_index) if metadata else []

        idx = -1
        for idx, sample in enumerate(tqdm(samples, position=pid or 0)):
            # index within the currently open file. Not to be confused with the index over a dataset
            # which may consist of multiple files.
            index = start_index + idx
            with db.begin(write=True) as txn:
                lmdb_put(txn, str(index), sample)
            for meta in metadata:
                meta.update(index, sample)

        if idx == -1:
            return 0

        # Save length and other info in lmdb
        length = idx + 1
        original_length = get_length(db)
        assert original_length == start_index
        with db.begin(write=True) as txn:
            lmdb_put(txn, "length", start_index + length)
        check_and_put_metadata(db, metadata, start_index + length)

        db.sync()

    return length


def check_and_init_metadata(
    db: lmdb.Environment,
    metadata: list[Metadata],
    length: int,
    return_all: bool = True,
    verbose: bool = False,
) -> list[Metadata]:
    new_metadata = []
    for meta in metadata:
        with db.begin() as txn:
            stored_value = lmdb_get(txn, meta.name, raise_if_missing=False)
        if stored_value is not None:
            if meta.is_frozen:
                if verbose:
                    print(f"stored value for {meta.name} is {stored_value}")
                assert (
                    meta.value == stored_value
                ), f"Expected metadata {meta.name} to have value {meta.value}, but got {stored_value} in database {db}."
            else:
                if verbose:
                    print(f"checking {meta.name}")
                # if the metadata is not frozen, reinitialize with the stored value.
                meta = meta.from_value(value=stored_value)
                # Check consistency.
                meta.check(length)
        if stored_value is None or return_all:
            new_metadata.append(meta)
    return new_metadata


def check_and_put_metadata(db: lmdb.Environment, metadata: list[Metadata], length: int):
    for meta in metadata:
        meta.check(length)
        with db.begin(write=True) as txn:
            lmdb_put(txn, meta.name, meta.value)


def ensure_metadata(db_path: str, metadata: list[Metadata]) -> int:
    """Checks the metadata values stored in the database and compute missing ones to
    ensure that all metadata are present.

    Args:
        db_path: path to the database .lmdb file.
        metadata: list of metadata.

    Returns:
        the length of the database.
    """
    with lmdb_open(db_path) as db:

        length = get_length(db)

        metadata = check_and_init_metadata(db, metadata, length, return_all=False)  # only new ones

        if len(metadata) == 0:
            # No metadata to add
            return 0

        print(f"Need to compute missing metadata {[m.name for m in metadata]}")

        for i in range(length):
            with db.begin() as txn:
                sample = lmdb_get(txn, str(i))
            for meta in metadata:
                meta.update(i, sample)

        check_and_put_metadata(db, metadata, length)

        db.sync()

    return length
