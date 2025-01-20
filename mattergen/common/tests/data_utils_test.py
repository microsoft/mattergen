# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Adapted from https://github.com/txie-93/cdvae/blob/main/cdvae/common/data_utils_test.py.
# Published under MIT license: https://github.com/txie-93/cdvae/blob/main/LICENSE.

from collections import Counter
from itertools import product
from typing import Dict, Optional, Tuple

import numpy as np
import pytest
import torch
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import RotationTransformation

from mattergen.common.tests.testutils import get_mp_20_debug_batch
from mattergen.common.utils import data_utils


def test_lattice_params_matrix():
    a, b, c = 4.0, 3.0, 2.0
    alpha, beta, gamma = 120.0, 90.0, 90.0

    matrix = data_utils.lattice_params_to_matrix(a, b, c, alpha, beta, gamma)
    result = data_utils.lattice_matrix_to_params(matrix)

    assert np.allclose([a, b, c, alpha, beta, gamma], result)


def test_lattice_params_matrix2():
    matrix = [
        [3.96686600e00, 0.00000000e00, 2.42900487e-16],
        [-2.42900487e-16, 3.96686600e00, 2.42900487e-16],
        [0.00000000e00, 0.00000000e00, 5.73442000e00],
    ]
    matrix = np.array(matrix)
    params = data_utils.lattice_matrix_to_params(matrix)
    result = data_utils.lattice_params_to_matrix(*params)

    assert np.allclose(matrix, result)


def test_lattice_params_to_matrix_torch():
    lengths = np.array([[4.0, 3.0, 2.0], [1, 3, 2]])
    angles = np.array([[120.0, 90.0, 90.0], [57.0, 130.0, 85.0]])

    lengths_and_angles = np.concatenate([lengths, angles], axis=-1)

    matrix0 = data_utils.lattice_params_to_matrix(*lengths_and_angles[0].tolist())
    matrix1 = data_utils.lattice_params_to_matrix(*lengths_and_angles[1].tolist())

    true_matrix = np.stack([matrix0, matrix1], axis=0)

    torch_matrix = data_utils.lattice_params_to_matrix_torch(
        torch.Tensor(lengths), torch.Tensor(angles)
    )

    assert np.allclose(true_matrix, torch_matrix.numpy(), atol=1e-5)


def test_lattice_matrix_to_params_torch():
    lengths = np.array([[4.0, 3.0, 2.0], [1, 3, 2]])
    angles = np.array([[120.0, 90.0, 90.0], [57.0, 130.0, 85.0]])

    torch_matrix = data_utils.lattice_params_to_matrix_torch(
        torch.Tensor(lengths), torch.Tensor(angles)
    )
    torch_lengths, torch_angles = data_utils.lattice_matrix_to_params_torch(torch_matrix)
    assert np.allclose(lengths, torch_lengths.numpy(), atol=1e-5)
    assert np.allclose(angles, torch_angles.numpy(), atol=1e-5)


def test_frac_cart_conversion():
    num_atoms = torch.LongTensor([4, 3, 2, 5])
    lengths = torch.rand(num_atoms.size(0), 3) * 4
    angles = torch.rand(num_atoms.size(0), 3) * 60 + 60
    frac_coords = torch.rand(num_atoms.sum(), 3)

    cart_coords = data_utils.frac_to_cart_coords(frac_coords, lengths, angles, num_atoms)

    inverted_frac_coords = data_utils.cart_to_frac_coords(cart_coords, lengths, angles, num_atoms)

    assert torch.allclose(frac_coords, inverted_frac_coords, atol=1e-5, rtol=1e-3)


def test_get_pbc_distances():
    frac_coords = torch.Tensor([[0.2, 0.2, 0.0], [0.6, 0.8, 0.8], [0.2, 0.2, 0.0], [0.6, 0.8, 0.8]])
    edge_index = torch.LongTensor([[1, 0], [0, 0], [2, 3]]).T
    lengths = torch.Tensor([[1.0, 1.0, 2.0], [1.0, 2.0, 1.0]])
    angles = torch.Tensor([[90.0, 90.0, 90.0], [90.0, 90.0, 90.0]])
    to_jimages = torch.LongTensor([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
    num_nodes = torch.LongTensor([2, 2])
    num_edges = torch.LongTensor([2, 1])

    lattice = data_utils.lattice_params_to_matrix_torch(lengths, angles)
    out = data_utils.get_pbc_distances(
        frac_coords, edge_index, lattice, to_jimages, num_nodes, num_edges
    )

    true_distances = torch.Tensor([1.7549928774784245, 1.0, 1.2])

    assert torch.allclose(true_distances, out["distances"])


def test_get_pbc_distances_cart():
    frac_coords = torch.Tensor([[0.2, 0.2, 0.0], [0.6, 0.8, 0.8], [0.2, 0.2, 0.0], [0.6, 0.8, 0.8]])
    edge_index = torch.LongTensor([[1, 0], [0, 0], [2, 3]]).T
    lengths = torch.Tensor([[1.0, 1.0, 2.0], [1.0, 2.0, 1.0]])
    angles = torch.Tensor([[90.0, 90.0, 90.0], [90.0, 90.0, 90.0]])
    to_jimages = torch.LongTensor([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
    num_nodes = torch.LongTensor([2, 2])
    num_edges = torch.LongTensor([2, 1])

    cart_coords = data_utils.frac_to_cart_coords(frac_coords, lengths, angles, num_nodes)

    lattice = data_utils.lattice_params_to_matrix_torch(lengths, angles)
    out = data_utils.get_pbc_distances(
        cart_coords,
        edge_index,
        lattice,
        to_jimages,
        num_nodes,
        num_edges,
        coord_is_cart=True,
    )

    true_distances = torch.Tensor([1.7549928774784245, 1.0, 1.2])

    assert torch.allclose(true_distances, out["distances"])


@pytest.mark.parametrize(
    "max_radius,max_neighbors",
    [
        (5.5964, 100),
        (5.6, 100),
        (100.0, 100),
        (7.0, 14),
        (7.0, 15),
    ],
)
def test_pbc_graph_translation_invariant(max_radius: float, max_neighbors: int):
    # if we perform in 32 bit for (max_radius, max_neighbors) = (5.596532197709578, 100)
    # simple cubic lattice
    lengths = torch.tensor([4.0, 4.0, 4.0])[None, :]
    angles = torch.tensor([90.0, 90.0, 90.0])[None, :]
    frac_coords = torch.tensor([[0.2, 0.0, 0.0], [0.9927, 0.5, 0.5]])
    num_atoms = torch.tensor([2])

    cart_coords = data_utils.frac_to_cart_coords(frac_coords, lengths, angles, num_atoms)
    # global translation, which should not affect the neighbor graph
    translation = torch.tensor([[0.05, 0.1, -0.04]])
    cart_coords_translated = cart_coords + translation
    frac_coords_translated = data_utils.cart_to_frac_coords(
        cart_coords_translated, lengths, angles, num_atoms
    )
    cart_coords_translated = data_utils.frac_to_cart_coords(
        frac_coords_translated, lengths, angles, num_atoms
    )

    lattice = data_utils.lattice_params_to_matrix_torch(lengths=lengths, angles=angles)

    coords = {"original": cart_coords, "translated": cart_coords_translated}

    # mypy complains without this type annotation
    output: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {
        coord: {
            output_type: {
                max_cells: {c: torch.tensor([0]) for c in coords.keys()} for max_cells in [1, 2]
            }
            for output_type in ["edge_index", "to_jimages", "num_bonds"]
        }
        for coord in coords.keys()
    }

    for coord in coords.keys():
        for max_cells in [2, 3]:
            (
                output[coord]["edge_index"][max_cells],
                output[coord]["to_jimages"][max_cells],
                output[coord]["num_bonds"][max_cells],
            ) = data_utils.radius_graph_pbc(
                cart_coords=coords[coord],
                lattice=lattice,
                num_atoms=num_atoms,
                radius=max_radius,
                max_num_neighbors_threshold=max_neighbors,
                max_cell_images_per_dim=max_cells,
            )

    for max_cell in [2, 3]:
        # max_cell>2 should be fine for this system
        counter1 = Counter(
            [tuple(x) for x in output["original"]["edge_index"][max_cell].t().tolist()]
        )
        counter2 = Counter(
            [tuple(x) for x in output["translated"]["edge_index"][max_cell].t().tolist()]
        )
        assert counter1 == counter2
        assert torch.equal(
            output["original"]["num_bonds"][max_cell], output["translated"]["num_bonds"][max_cell]
        )


def get_random_rotation(
    n_random: int, n_atom: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # randomly generate lattice
    lattice = torch.normal(mean=0, std=1, size=(3, 3))

    if n_atom is None:
        # uniformly pick number of atoms between [1,16] inclusive
        number_atoms = torch.randint(1, 17, (1,))
    else:
        number_atoms = torch.tensor([n_atom])

    # shape=[number_atoms, 3], ~U[0,1]
    frac_coord = torch.rand(size=(number_atoms[0], 3))

    structure = Structure(
        species=["C" for _ in range(number_atoms[0])],
        lattice=lattice.numpy(),
        coords=frac_coord.numpy(),
    )

    # rotation axies respect to cell vectors: np.ndarray, shape=[n_random, 3], dtype=int in [0, 1]
    random_axes = np.random.choice([0, 1], size=(n_random, 3))

    for ii, axis in enumerate(random_axes):
        if np.allclose(axis, [0, 0, 0]):
            # will get nans if we try to rotate along (0,0,0) axis
            random_axes[ii] = [1, 0, 0]

    # random rotation angles between [0,90] degrees, shape=(n_random,)
    random_angles = np.random.rand(n_random) * 90

    # List[pymatgen.core.structure.Structure], shape=(n_random,)
    structures = [
        RotationTransformation(axis=axis, angle=angle).apply_transformation(structure)
        for (axis, angle) in zip(random_axes, random_angles)
    ]

    # shape=(n_random, 3, 3)
    lattices = torch.tensor(
        np.asarray([s.lattice._matrix for s in structures]), dtype=torch.float32
    )

    # frac coords are constant under rotation, shape=(n_random*Natom, 3)
    frac_coords = (
        torch.tensor(structure.frac_coords, dtype=torch.float32)
        .expand((n_random, number_atoms[0], 3))
        .flatten(end_dim=1)
    )

    # shape=(n_random, )
    num_atoms = torch.tensor([structure.frac_coords.shape[0]]).expand(n_random)

    return (
        data_utils.frac_to_cart_coords_with_lattice(
            frac_coords=frac_coords, lattice=lattices, num_atoms=num_atoms
        ),
        lattices,
        num_atoms,
    )


def get_random_translation(n_random: int, n_atom: Optional[int] = None):
    # randomly generate lattice
    lattice = torch.normal(mean=0, std=0.5, size=(3, 3))

    # make sure lattice is not incredibly skew as we could need an infinite number of
    # periodic cells in principle
    lattice[torch.eye(3).byte()] = torch.normal(mean=0, std=1, size=(3,))

    if n_atom is None:
        # uniformly pick number of atoms between [1,16] inclusive
        number_atoms = torch.randint(1, 17, (1,))
    else:
        number_atoms = torch.tensor([n_atom])

    # shape=[number_atoms, 3], ~U[0,1]
    frac_coord = torch.rand(size=(number_atoms[0], 3))

    # number of atom in each crystal, shape=[n_random]
    natoms = torch.tensor([frac_coord.shape[0]]).expand(n_random)

    # shape=[n_random, 3, 3]
    multiple_lattices = lattice.expand([n_random, 3, 3])

    # shape=[n_random, Natm, 3]
    translation = torch.rand(size=(n_random, 1, 3)).expand((n_random, frac_coord.shape[0], 3))

    # shape=[n_random, Natm, 3]
    new_frac_coord = frac_coord.expand((n_random, frac_coord.shape[0], 3)) + translation

    # ensure all fractional coordinates are between [0,1] inclusive
    new_frac_coord = new_frac_coord % 1

    # shape=[n_random*Natm, 3]
    new_frac_coord = new_frac_coord.flatten(end_dim=1)

    # map back to within unit cell and make cartesian, shape=[n_random*Natm, 3]
    new_cart_coord = data_utils.frac_to_cart_coords_with_lattice(
        frac_coords=new_frac_coord, lattice=multiple_lattices, num_atoms=natoms
    )

    return new_cart_coord, multiple_lattices, natoms


def check_invariance(
    max_radius: float,
    max_cell_images_per_dim: int,
    cart: torch.Tensor,
    lattice: torch.Tensor,
    num_atoms: torch.Tensor,
):
    # cart.shape=(Ncrystals*Natoms, 3)
    # lattice.shape=(Ncrystals, 3, 3)
    # num_atoms.shape=(Ncrystals,)

    max_neighbors = 100

    edges, _, num_bonds = data_utils.radius_graph_pbc(
        cart_coords=cart,
        lattice=lattice,
        num_atoms=num_atoms,
        radius=max_radius,
        max_num_neighbors_threshold=max_neighbors,
        max_cell_images_per_dim=max_cell_images_per_dim,
    )

    edges = edges.numpy()

    # group bonds by crystal
    start_from = np.asarray(np.hstack((np.zeros(1), np.cumsum(num_bonds))), dtype=int)

    counters = []
    for ii in range(len(start_from) - 1):
        # transpose shape to [Nbonds, 2]
        bond_subset = edges.T[start_from[ii] : start_from[ii + 1]]

        # bond indices are cumulatice over crystals
        offset = num_atoms[0] * ii

        # ensure all atom indices are 0-offset in a single crystal
        bond_subset -= offset.numpy()

        # print(len(bond_subset))

        # counter for bond pairs
        counters.append(Counter([tuple(x) for x in bond_subset]))

    # convert Counter to str so can hash and count
    count_counters = Counter([f"{c}" for c in counters])

    assert len(set([len(c) for c in counters])) == 1, set([len(c) for c in counters])
    assert len(count_counters) == 1, count_counters


@pytest.mark.parametrize(
    "max_radius, max_cell_images",
    [
        (3.0, 1),
        (7.0, 1),
        (3.0, 2),
        (7.0, 2),
        (3.0, 3),
        (7.0, 3),
    ],
)
def test_rotation_invariance(max_radius: float, max_cell_images: int):
    cart, lattice, num_atoms = get_random_rotation(n_random=10)
    check_invariance(
        max_radius=max_radius,
        max_cell_images_per_dim=max_cell_images,
        cart=cart,
        lattice=lattice,
        num_atoms=num_atoms,
    )


@pytest.mark.parametrize(
    "max_radius, max_cell_images",
    [
        (3.0, 10),  # we have random lattice matrices so need a generous number of max cell images
        (7.0, 20),
    ],
)
def test_translation_invariance(max_radius: float, max_cell_images: int):
    cart, lattice, num_atoms = get_random_translation(n_random=10)
    check_invariance(
        max_radius=max_radius,
        max_cell_images_per_dim=max_cell_images,
        cart=cart,
        lattice=lattice,
        num_atoms=num_atoms,
    )


def get_distances_pymatgen(structure: Structure, rcut: float) -> np.ndarray:
    neigh = structure.get_all_neighbors(r=rcut, include_image=True)
    dist = sorted(
        np.asarray([n.nn_distance for _atom in neigh for n in _atom if n.nn_distance > 1e-12])
    )
    return np.asarray(dist)


def get_distance_pytorch(structure: Structure, rcut: float) -> np.ndarray:
    cart_coords = torch.tensor(structure.cart_coords, dtype=torch.float32)
    lattice = torch.tensor([structure.lattice._matrix], dtype=torch.float32)
    num_atoms = torch.tensor([cart_coords.shape[0]], dtype=torch.int32)

    edges, images, num_bonds = data_utils.radius_graph_pbc(
        cart_coords=cart_coords,
        lattice=lattice,
        num_atoms=num_atoms,
        radius=rcut,
        max_num_neighbors_threshold=100000,
        max_cell_images_per_dim=100,
    )

    distances = data_utils.get_pbc_distances(
        coords=cart_coords,
        edge_index=edges,
        lattice=lattice,
        to_jimages=images,
        num_atoms=num_atoms,
        num_bonds=num_bonds,
        coord_is_cart=True,
    )

    return np.asarray(sorted(distances["distances"].numpy()))


def get_distances_numpy(structure: Structure, rcut: float, dtype) -> np.ndarray:
    # returns 1-d sorted np.ndarray of distances
    # lattice[i][x] is the xth cartesian component of lattice vector i
    # frac_coord[n][i] is the fractional coordinates of atom n with respect to lattice i

    frac_coord = np.asarray(structure.frac_coords, dtype=dtype)
    lattice = np.asarray(structure.lattice._matrix, dtype=dtype)

    natm = frac_coord.shape[0]

    # shape=(natm, 3)
    cart_coord_0_0_0 = np.asarray(np.einsum("ni, ix->nx", frac_coord, lattice), dtype=dtype)

    # this should be generously large
    max_cell = 100

    # shape = (nimages, 3)
    images = np.asarray(
        list(
            product(
                range(-max_cell, max_cell + 1),
                range(-max_cell, max_cell + 1),
                range(-max_cell, max_cell + 1),
            )
        ),
        dtype=dtype,
    )

    nimages = images.shape[0]

    # shape = (nimages, natoms, 3)
    images = np.tile(np.expand_dims(images, 1), (1, natm, 1))

    # shape = (nimages, natoms, 3)
    periodic_frac_coord = np.tile(frac_coord, (nimages, 1, 1)) + images

    # shape = (natm, nimages, natoms, 3)
    periodic_frac_coord = np.tile(np.expand_dims(periodic_frac_coord, 0), (natm, 1, 1, 1))

    assert periodic_frac_coord.dtype == dtype

    # shape = (natm, nimages, natoms, 3)
    cart_coords_tiled = np.tile(np.expand_dims(cart_coord_0_0_0, (1, 2)), (1, nimages, natm, 1))

    # shape = (natm, nimages, natm, 3)
    periodic_cart_coord = np.einsum("nimk,kx->nimx", periodic_frac_coord, lattice)

    assert periodic_cart_coord.dtype == dtype

    # shape = (natm, nimages, natm)
    all_distances = np.linalg.norm(cart_coords_tiled - periodic_cart_coord, axis=-1)

    # shape = (natm**2 * nimages)
    all_distances = all_distances.flatten()

    # discard zero distances (atom self interaction in same cell) and large distances
    all_distances = all_distances[
        np.where(np.logical_and(all_distances <= rcut, all_distances > 1e-12))[0]
    ]
    assert all_distances.dtype == dtype
    return np.asarray(sorted(all_distances))


@pytest.mark.parametrize(
    "natom, rcut",
    [
        (1, 1.0),
        (2, 1.0),
        (3, 1.0),
        (1, 2.0),
        (2, 2.0),
        (3, 2.0),
    ],
)
def test_rdf(natom: int, rcut: float):
    # random structure with lattice~N(0,1) and uniform random fractional coords
    structure = Structure(
        species=["C" for _ in range(natom)],
        coords=np.random.uniform(size=(natom, 3)),
        lattice=np.random.normal(size=(3, 3)),
    )
    assert np.allclose(
        get_distances_numpy(structure=structure, rcut=rcut, dtype=np.float32),
        get_distance_pytorch(structure=structure, rcut=rcut),
    )


def test_polar_decomposition():
    # load some data
    batch = get_mp_20_debug_batch()
    lattices = data_utils.lattice_params_to_matrix_torch(batch.lengths, batch.angles)
    polar_decomposition = data_utils.compute_lattice_polar_decomposition(lattices)
    symm_lengths, symm_angles = data_utils.lattice_matrix_to_params_torch(polar_decomposition)
    assert torch.allclose(symm_lengths, batch.lengths, atol=1e-3)
    assert torch.allclose(symm_angles, batch.angles, atol=1e-3)
    assert torch.allclose(polar_decomposition.det().abs(), lattices.det().abs(), atol=1e-3)


def test_torch_nanstd():
    x = torch.tensor([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, np.nan, 6.0])
    assert data_utils.torch_nanstd(x=x, dim=0, unbiased=False).item() == np.nanstd(x.numpy())
