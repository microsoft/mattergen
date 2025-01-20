# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Iterable, Protocol, Sequence, Type, TypeVar

import numpy as np
import numpy.typing
import pandas as pd
import torch
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.groups import SpaceGroup
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.transform import Transform
from mattergen.common.data.types import PropertySourceId, PropertyValues
from mattergen.common.globals import PROJECT_ROOT
from mattergen.common.utils.globals import PROPERTY_SOURCE_IDS

CORE_STRUCTURE_FILE_NAMES = {
    "pos": "pos.npy",
    "cell": "cell.npy",
    "atomic_numbers": "atomic_numbers.npy",
    "num_atoms": "num_atoms.npy",
    "structure_id": "structure_id.npy",
}

T = TypeVar("T", bound="BaseDataset")


class DatasetTransform(Protocol):
    def __call__(self, dataset: "BaseDataset") -> "BaseDataset": ...


@lru_cache
def space_group_number_for_symbol(symbol: str) -> int:
    return SpaceGroup(symbol).int_number


@dataclass(frozen=True)
class BaseDataset(Dataset):
    properties: dict[PropertySourceId, numpy.typing.NDArray]

    def __getitem__(self, index: int) -> ChemGraph:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def get_properties_dict(self, index: int) -> dict[PropertySourceId, torch.Tensor]:
        props_dict: dict[PropertySourceId, torch.Tensor] = {}
        for prop in self.properties.keys():
            if prop == "chemical_system":
                # chemical system is set via a data transform
                continue
            val = self.properties[prop][index]
            if prop == "space_group":
                val = space_group_number_for_symbol(val)
            props_dict[prop] = (
                torch.from_numpy(val) if isinstance(val, np.ndarray) else torch.tensor(val)
            )
            if props_dict[prop].dtype == torch.double:
                props_dict[prop] = props_dict[prop].float()
        return props_dict

    @classmethod
    def from_dataset_name(
        cls: Type[T],
        dataset_name: str,
        split: str,
        transforms: list[Transform] | None = None,
        properties: list[PropertySourceId] | None = None,
        dataset_transforms: list[DatasetTransform] | None = None,
    ):
        """
        Load a dataset using a dataset name and split. We assume the dataset is stored in the
        datasets folder in the project root.
        """
        return CrystalDatasetBuilder.from_dataset_name(
            dataset_name=dataset_name,
            split=split,
            transforms=transforms,
            properties=properties,
        ).build(cls, dataset_transforms=dataset_transforms)

    @classmethod
    def from_cache_path(
        cls: Type[T],
        cache_path: str,
        transforms: list[Transform] | None = None,
        properties: list[PropertySourceId] | None = None,
        dataset_transforms: list[DatasetTransform] | None = None,
    ) -> T:
        """
        Load a dataset from a specified cache path.

        Args:
            name: Name of the reference dataset.
            transforms: List of transforms to apply to **each datapoint** when loading, e.g., to make the lattice matrices symmetric.
            properties: List of properties to condition on.
            dataset_transforms: List of transforms to apply to the **whole dataset**, e.g., to filter out certain entries.

        Returns:
            The dataset.
        """
        return CrystalDatasetBuilder.from_cache_path(
            cache_path=cache_path,
            transforms=transforms,
            properties=properties,
        ).build(cls, dataset_transforms=dataset_transforms)

    def subset(self, indices: Sequence[int]) -> "BaseDataset":
        """
        Create a subset of the dataset with the given indices.
        """
        raise NotImplementedError

    def repeat(self, repeats: int) -> "BaseDataset":
        """
        Repeat the dataset a number of times.
        """
        raise NotImplementedError


def repeat_along_first_axis(
    input_array: numpy.typing.NDArray, repeats: int
) -> numpy.typing.NDArray:
    # np.tile by default repeats along the last axis. So we need to pass a tuple
    # with the number of repeats for each axis, e.g., (repeats, 1, 1) for the cell.
    return np.tile(input_array, (repeats,) + tuple(np.ones(input_array.ndim - 1, dtype=int)))


@dataclass(frozen=True, kw_only=True)
class CrystalDataset(BaseDataset):
    """
    Dataset for crystal structures. Takes as input numpy arrays for positions, cell, atomic numbers,
    number of atoms and structure id. Optionally, properties can be added as well, as a dictionary
    of numpy arrays. The dataset can also be transformed using a list of transforms.
    The recommended way of creating a CrystalDataset is to use the class method
    CrystalDataset.from_preset with a preset name, which will use the CrystalDatasetBuilder class to
    fetch the dataset from cache if it exists, and otherwise cache it.
    """

    pos: numpy.typing.NDArray
    cell: numpy.typing.NDArray
    atomic_numbers: numpy.typing.NDArray
    num_atoms: numpy.typing.NDArray
    structure_id: numpy.typing.NDArray
    properties: dict[PropertySourceId, numpy.typing.NDArray] = field(default_factory=dict)
    transforms: list[Transform] | None = None

    def __post_init__(self):
        property_names = list(self.properties.keys())
        assert all([s in PROPERTY_SOURCE_IDS for s in property_names]), (
            f"Property names {property_names} are not valid. "
            f"Valid property source names: {PROPERTY_SOURCE_IDS}"
        )

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        cache_path: str,
        transforms: list[Transform] | None = None,
    ):
        return CrystalDatasetBuilder.from_csv(
            csv_path=csv_path,
            cache_path=cache_path,
            transforms=transforms,
        ).build(cls)

    @cached_property
    def index_offset(self):
        """
        Returns an array of indices that can be used to offset the indices of the atoms.
        That is, for structure index <ix>, the atoms are located at indices
        <index_offset[ix]:index_offset[ix]+num_atoms[ix]> in the pos and atomic_numbers arrays.
        """
        return np.concatenate([np.array([0]), np.cumsum(self.num_atoms[:-1])])

    def __getitem__(self, index: int) -> ChemGraph:
        pos_offset = self.index_offset[index]
        num_atoms = torch.tensor(self.num_atoms[index])

        props_dict = self.get_properties_dict(index)
        data = ChemGraph(
            pos=torch.from_numpy(self.pos[pos_offset : pos_offset + num_atoms]).float() % 1.0,
            cell=torch.from_numpy(self.cell[index]).float().unsqueeze(0),
            atomic_numbers=torch.from_numpy(
                self.atomic_numbers[pos_offset : pos_offset + num_atoms]
            ),
            num_atoms=num_atoms,
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            # mypy does not like string literals as kwargs, see https://github.com/python/mypy/pull/10237
            **props_dict,  # type: ignore
        )

        if self.transforms is not None:
            for t in self.transforms:
                data = t(data)
        return data

    def __len__(self) -> int:
        return len(self.num_atoms)

    def subset(self, indices: Sequence[int]) -> "CrystalDataset":
        batch_indices: list[int] = []
        for index in indices:
            pos_offset = self.index_offset[index]
            batch_indices.extend(range(pos_offset, pos_offset + self.num_atoms[index]))

        return CrystalDataset(
            pos=self.pos[batch_indices],
            cell=self.cell[indices],
            atomic_numbers=self.atomic_numbers[batch_indices],
            num_atoms=self.num_atoms[indices],
            structure_id=self.structure_id[indices],
            properties={k: v[indices] for k, v in self.properties.items()},
            transforms=self.transforms,
        )

    def repeat(self, repeats: int) -> "CrystalDataset":
        """
        Repeat the dataset a number of times.
        """

        pos = repeat_along_first_axis(self.pos, repeats)
        cell = repeat_along_first_axis(self.cell, repeats)
        atomic_numbers = repeat_along_first_axis(self.atomic_numbers, repeats)
        num_atoms = repeat_along_first_axis(self.num_atoms, repeats)
        structure_id = repeat_along_first_axis(self.structure_id, repeats)
        properties = {k: repeat_along_first_axis(v, repeats) for k, v in self.properties.items()}
        return CrystalDataset(
            pos=pos,
            cell=cell,
            atomic_numbers=atomic_numbers,
            num_atoms=num_atoms,
            structure_id=structure_id,
            properties=properties,
            transforms=self.transforms,
        )


@dataclass(frozen=True, kw_only=True)
class NumAtomsCrystalDataset(BaseDataset):
    """
    A dataset class for crystal structures where the number of atoms is the only property. Optionally,
    other properties can be added as well, as a dictionary of numpy arrays.
    This is useful for sampling, where only need to condition on the number of atoms in the structure.
    Positions and cell are filled with NaNs, and the atomic numbers are filled with -1 for ChemGraphs
    that are created from this dataset.
    """

    num_atoms: numpy.typing.NDArray
    structure_id: numpy.typing.NDArray | None = None
    properties: dict[PropertySourceId, numpy.typing.NDArray] = field(default_factory=dict)
    transforms: list[Transform] | None = None

    def __getitem__(self, index: int) -> ChemGraph:
        num_atoms = torch.tensor(self.num_atoms[index])

        props_dict = self.get_properties_dict(index)
        data = ChemGraph(
            pos=torch.full((num_atoms, 3), fill_value=torch.nan, dtype=torch.float),
            cell=torch.full((1, 3, 3), fill_value=torch.nan, dtype=torch.float),
            atomic_numbers=torch.full((num_atoms,), fill_value=-1, dtype=torch.long),
            num_atoms=num_atoms,
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            # mypy does not like string literals as kwargs, see https://github.com/python/mypy/pull/10237
            **props_dict,  # type: ignore
        )

        if self.transforms is not None:
            for t in self.transforms:
                data = t(data)
        return data

    def __len__(self) -> int:
        return len(self.num_atoms)

    def subset(self, indices: Sequence[int]) -> "NumAtomsCrystalDataset":
        return NumAtomsCrystalDataset(
            num_atoms=self.num_atoms[indices],
            structure_id=self.structure_id[indices] if self.structure_id is not None else None,
            properties={k: v[indices] for k, v in self.properties.items()},
            transforms=self.transforms,
        )

    def repeat(self, repeats: int) -> "NumAtomsCrystalDataset":
        """
        Repeat the dataset a number of times.
        """
        num_atoms = repeat_along_first_axis(self.num_atoms, repeats)
        structure_id = repeat_along_first_axis(self.structure_id, repeats)
        properties = {k: repeat_along_first_axis(v, repeats) for k, v in self.properties.items()}
        return NumAtomsCrystalDataset(
            num_atoms=num_atoms,
            structure_id=structure_id,
            properties=properties,
            transforms=self.transforms,
        )

    @classmethod
    def from_num_atoms_distribution(
        cls: Type[T],
        num_atoms_distribution: dict[int, float],
        num_samples: int,
        transforms: list[Transform] | None = None,
    ) -> T:
        """
        Construct a NumAtomsCrystalDataset from a distribution over number of atoms.

        Args:
            num_atoms_distribution: A dictionary with the number of atoms as keys and the probability of that number of atoms as values.
            transforms: List of transforms to apply to **each datapoint** when loading, e.g., to make the lattice matrices symmetric.
            properties: List of properties to condition on.
            dataset_transforms: List of transforms to apply to the **whole dataset**, e.g., to filter out certain entries.

        Returns:
            The dataset.
        """

        return NumAtomsCrystalDataset(
            num_atoms=np.random.choice(
                list(num_atoms_distribution.keys()),
                size=num_samples,
                p=list(num_atoms_distribution.values()),
            ),
            transforms=transforms,
        )


def structures_to_numpy(
    structures: Iterable[Structure],
) -> tuple[dict[str, numpy.typing.NDArray], dict[PropertySourceId, numpy.typing.NDArray]]:
    """
    Convert a list of Structures to numpy arrays for positions, cell, atomic numbers,
    number of atoms and structure id. Returns a dictionary with the numpy arrays.
    """
    structure_infos: dict[str, list[numpy.typing.NDArray]] = {
        "pos": [],
        "cell": [],
        "atomic_numbers": [],
        "num_atoms": [],
        "structure_id": [],
    }
    properties = defaultdict(list)
    for structure in tqdm(structures, desc="Converting structures to numpy", miniters=5000):
        # get primitive structure
        # here, structure.properties is not passed to struct if it is not a primitive structure,
        # so we keep the structure object to pass material_id below
        struct = structure.get_primitive_structure()
        # niggli reduction
        struct = struct.get_reduced_structure()

        structure_infos["pos"].append(struct.frac_coords)
        structure_infos["cell"].append(struct.lattice.matrix)
        structure_infos["atomic_numbers"].append(struct.atomic_numbers)
        structure_infos["num_atoms"].append(len(struct))
        structure_infos["structure_id"].append(structure.properties["material_id"])
        for prop, prop_val in structure.properties.items():
            if prop in PROPERTY_SOURCE_IDS:
                properties[prop].append(prop_val)
    structure_infos["pos"] = np.row_stack(structure_infos["pos"])
    structure_infos["cell"] = np.array(structure_infos["cell"])
    structure_infos["atomic_numbers"] = np.concatenate(structure_infos["atomic_numbers"])
    structure_infos["num_atoms"] = np.array(structure_infos["num_atoms"])
    structure_infos["structure_id"] = np.array(structure_infos["structure_id"])
    for prop in properties:
        properties[prop] = np.array(properties[prop])
        assert len(properties[prop]) == len(structure_infos["structure_id"])
    return structure_infos, properties


class CrystalDatasetBuilder:
    """
    Class for building CrystalDatasets. The builder handles the caching of the numpy arrays and
    properties, and can be used to add new properties to the cache.

    The most common way to use the CrystalDatasetBuilder is to use the from_preset method, which
    only requires the name of the reference dataset. The builder will then check if the dataset is
    already cached, and if not, cache it. The builder can also be used to add new properties to the
    cache.
    """

    def __init__(
        self,
        cache_path: str,
        transforms: list[Transform] | None = None,
        properties: list[PropertySourceId] | None = None,
    ):
        self.cache_path = cache_path
        self.transforms = transforms
        self.property_names = properties or []
        assert all([s in PROPERTY_SOURCE_IDS for s in self.property_names]), (
            f"Property names {self.property_names} are not valid. "
            f"Valid property source names: {PROPERTY_SOURCE_IDS}"
        )

    def _load_file(self, filename: str) -> numpy.typing.NDArray:
        return np.load(f"{self.cache_path}/{filename}")

    @cached_property
    def pos(self):
        return self._load_file(CORE_STRUCTURE_FILE_NAMES["pos"])

    @cached_property
    def cell(self):
        return self._load_file(CORE_STRUCTURE_FILE_NAMES["cell"])

    @cached_property
    def atomic_numbers(self):
        return self._load_file(CORE_STRUCTURE_FILE_NAMES["atomic_numbers"])

    @cached_property
    def num_atoms(self):
        return self._load_file(CORE_STRUCTURE_FILE_NAMES["num_atoms"])

    @cached_property
    def structure_id(self):
        return self._load_file(CORE_STRUCTURE_FILE_NAMES["structure_id"])

    @property
    def properties(self) -> dict[PropertySourceId, numpy.typing.NDArray]:
        properties: dict[PropertySourceId, numpy.typing.NDArray] = {}
        prop_names = self.property_names
        for prop_name in prop_names:
            if not os.path.exists(f"{self.cache_path}/{prop_name}.json"):
                raise FileNotFoundError(
                    f"{prop_name}.json does not exist in {self.cache_path}.\n"
                    f"Available properties: {self.list_available_properties()}"
                )
            properties[prop_name] = PropertyValues.from_json(
                f"{self.cache_path}/{prop_name}.json"
            ).values
            assert len(properties[prop_name]) == len(self.structure_id)
        return properties

    def build(
        self,
        dataset_class: Type[T] = CrystalDataset,
        dataset_transforms: list[DatasetTransform] | None = None,
    ) -> T:
        """
        Build a dataset from the cached numpy arrays and properties. The dataset class can be
        either CrystalDataset, CrystalStructurePredictionSamplingDataset, or NumAtomsCrystalDataset.

        Args:
            dataset_class: The class of the dataset to build.
            dataset_transforms: List of transforms to apply to the dataset.
        """
        if dataset_class == CrystalDataset:
            dataset = self._build_full_dataset()
        elif dataset_class == NumAtomsCrystalDataset:
            dataset = self._build_num_atoms()
        else:
            raise ValueError(f"Unknown dataset class {dataset_class}.")
        dataset_transforms = dataset_transforms or []
        for t in dataset_transforms:
            dataset = t(dataset)
        return dataset

    def _build_full_dataset(self) -> CrystalDataset:
        """
        Build a CrystalDataset from the cached numpy arrays and properties.
        """

        dataset = CrystalDataset(
            pos=self.pos,
            cell=self.cell,
            atomic_numbers=self.atomic_numbers,
            num_atoms=self.num_atoms,
            structure_id=self.structure_id,
            properties=self.properties,
            transforms=self.transforms,
        )
        return dataset

    def _build_num_atoms(self) -> NumAtomsCrystalDataset:
        """
        Build a NumAtomsCrystalDataset from the cached numpy arrays and properties.
        """

        dataset = NumAtomsCrystalDataset(
            num_atoms=self.num_atoms,
            structure_id=self.structure_id,
            properties=self.properties,
            transforms=self.transforms,
        )
        return dataset

    @classmethod
    def from_dataset_name(
        cls,
        dataset_name: str,
        split: str,
        transforms: list[Transform] | None = None,
        properties: list[PropertySourceId] | None = None,
    ):
        return cls.from_cache_path(
            f"{PROJECT_ROOT}/datasets/{dataset_name}/{split}", transforms, properties
        )

    @classmethod
    def from_cache_path(
        cls,
        cache_path: str,
        transforms: list[Transform] | None = None,
        properties: list[PropertySourceId] | None = None,
    ) -> "CrystalDatasetBuilder":
        """
        Create a CrystalDatasetBuilder from a path that contains cache for the dataset.
        """

        return cls(
            cache_path=cache_path,
            transforms=transforms,
            properties=properties,
        )

    @classmethod
    def from_csv(cls, csv_path: str, cache_path: str, transforms: list[Transform] | None = None):
        df = pd.read_csv(csv_path)

        structures = [
            CifParser.from_str(s).parse_structures(primitive=True, on_error="ignore")[0]
            for s in tqdm(df["cif"], desc="Parsing CIFs", miniters=5000)
        ]
        for ix, material_id in enumerate(df["material_id"]):
            structures[ix].properties["material_id"] = material_id
            for prop in df.columns:
                if prop in PROPERTY_SOURCE_IDS:
                    structures[ix].properties[prop] = df[prop][ix]
        structure_infos, properties = structures_to_numpy(structures)

        os.makedirs(cache_path, exist_ok=True)
        print(f"Storing cached dataset in {cache_path}.")
        for k, filename in CORE_STRUCTURE_FILE_NAMES.items():
            np.save(f"{cache_path}/{filename}", structure_infos[k])
        for prop in properties:
            PropertyValues(
                values=properties[prop],
                property_source_doc_id=prop,
            ).to_json(f"{cache_path}/{prop}.json")

        return cls(
            cache_path=cache_path,
            transforms=transforms,
            properties=list(properties.keys()),
        )

    def list_available_properties(self) -> list[PropertySourceId]:
        """
        List the properties that are available in the cache.
        """
        return [
            prop.split(".json")[0] for prop in os.listdir(self.cache_path) if prop.endswith(".json")
        ]

    def add_property_to_cache(
        self,
        property_name: PropertySourceId,
        data: dict[str, numpy.typing.NDArray],
    ):
        """
        Add a new property to the cache. The property will be stored in the blob storage and added
        to the properties of the dataset.

        The data should be a dictionary with the structure id as keys and the property values as
        values. The properties can be sparse, i.e. some structures can be missing the property.
        These properties will be set to NaN in the dataset.
        """
        assert (
            property_name not in self.property_names
        ), f"Property {property_name} already exists in properties"
        property_values_linearized = np.array(
            [data.get(structure_id, np.nan) for structure_id in self.structure_id]
        )
        property_values = PropertyValues(
            values=property_values_linearized,
            property_source_doc_id=property_name,
        )
        assert property_values.n_entries == len(self.structure_id), (
            f"Property {property_name} has {property_values.n_entries} entries, "
            f"but the dataset has {len(self.structure_id)} structures."
        )
        property_values.to_json(self.cache_path + "/" + f"{property_name}.json")
        self.property_names.append(property_name)
