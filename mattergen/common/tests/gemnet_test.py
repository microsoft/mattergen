# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from copy import deepcopy
from itertools import chain, permutations
from typing import List, Tuple

import torch
from pymatgen.core.structure import Structure
from scipy.spatial.transform import Rotation
from torch_geometric.data import Batch, Data

from mattergen.common.gemnet.gemnet import GemNetT
from mattergen.common.gemnet.layers.embedding_block import AtomEmbedding
from mattergen.common.tests.testutils import get_mp_20_debug_batch
from mattergen.common.utils.data_utils import (
    cart_to_frac_coords_with_lattice,
    frac_to_cart_coords_with_lattice,
    lattice_matrix_to_params_torch,
    lattice_params_to_matrix_torch,
)
from mattergen.common.utils.eval_utils import make_structure
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT

### UTILS ###


def get_model(**kwargs) -> GemNetT:
    return GemNetT(
        atom_embedding=AtomEmbedding(emb_size=4),
        num_targets=1,
        latent_dim=4,
        num_radial=4,
        num_blocks=1,
        emb_size_atom=4,
        emb_size_edge=4,
        emb_size_trip=4,
        emb_size_bil_trip=4,
        otf_graph=True,
        scale_file=f"{MODELS_PROJECT_ROOT}/common/gemnet/gemnet-dT.json",
        **kwargs,
    )


def structures_list_to_batch(structures: List[Structure]) -> Batch:
    return Batch.from_data_list(
        [
            Data(
                angles=torch.tensor(s.lattice.angles, dtype=torch.float32)[None],
                lengths=torch.tensor(s.lattice.lengths, dtype=torch.float32)[None],
                frac_coords=torch.from_numpy(s.frac_coords).float(),
                atom_types=torch.tensor(s.atomic_numbers),
                num_atoms=s.num_sites,
                num_nodes=s.num_sites,
            )
            for s in structures
        ]
    )


def reformat_batch(
    batch: Batch,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    return (
        None,
        batch.frac_coords,
        batch.atom_types,
        batch.num_atoms,
        batch.batch,
        batch.lengths,
        batch.angles,
    )


def get_cubic_data(supercell: Tuple[int, int, int]) -> Tuple[Tuple, Tuple]:
    normal_structures = [
        Structure(lattice=[[2, 0, 0], [0, 3.1, 0], [0, 0, 2.9]], coords=[[0, 0, 0]], species="C"),
        Structure(
            lattice=[[3.1, 0, 0], [0, 2, 0], [0, 0, 4]],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            species=["C", "C"],
        ),
    ]
    # need a batch size of 64
    normal_structures = list(chain.from_iterable([deepcopy(normal_structures) for _ in range(32)]))

    supercell_structures = deepcopy(normal_structures)
    for s in supercell_structures:
        s.make_supercell(supercell)

    normal_batch = structures_list_to_batch(structures=normal_structures)
    supercell_batch = structures_list_to_batch(structures=supercell_structures)

    return reformat_batch(batch=normal_batch), reformat_batch(batch=supercell_batch)


### TESTS ###


def test_lattice_score_scale_invariance():
    # test invariance of lattice score to supercell size
    cutoff = 5.0
    max_neighbors = 1000
    torch.manual_seed(495606849)
    model = get_model(
        max_neighbors=max_neighbors,
        cutoff=cutoff,
        # regress stress in a non-conservative way
        regress_stress=True,
        max_cell_images_per_dim=20,
    )
    model.eval()
    batch = get_mp_20_debug_batch()
    # take a subset because the test is slow
    batch = Batch.from_data_list(batch.to_data_list()[:10])

    supercell_structures = [
        make_structure(d.lengths.squeeze(0), d.angles.squeeze(0), d.atom_types, d.frac_coords)
        for d in batch.to_data_list()
    ]
    for s in supercell_structures:
        s.make_supercell((2, 2, 2))
    supercell_batch = Batch.from_data_list(
        [
            Data(
                angles=torch.tensor(s.lattice.angles, dtype=torch.float32)[None],
                lengths=torch.tensor(s.lattice.lengths, dtype=torch.float32)[None],
                frac_coords=torch.from_numpy(s.frac_coords).float(),
                atom_types=torch.tensor(s.atomic_numbers),
                num_atoms=s.num_sites,
                num_nodes=s.num_sites,
            )
            for s in supercell_structures
        ]
    )

    with torch.no_grad():
        out_normal_cells = model.forward(
            None,
            batch.frac_coords,
            batch.atom_types,
            batch.num_atoms,
            batch.batch,
            batch.lengths,
            batch.angles,
        )

        out_supercells = model.forward(
            None,
            supercell_batch.frac_coords,
            supercell_batch.atom_types,
            supercell_batch.num_atoms,
            supercell_batch.batch,
            supercell_batch.lengths,
            supercell_batch.angles,
        )

    # for mypy
    assert out_normal_cells.stress is not None
    assert out_supercells.stress is not None

    all_close = torch.allclose(out_normal_cells.stress, out_supercells.stress, atol=1e-5)
    assert all_close, (out_normal_cells.stress - out_supercells.stress).abs().max()


def test_nonconservative_lattice_score_translation_invariance():
    model = get_model(
        max_neighbors=200,
        cutoff=5.0,
        regress_stress=True,
        max_cell_images_per_dim=10,
    )
    model.eval()
    batch = get_mp_20_debug_batch()

    structures = [
        make_structure(d.lengths.squeeze(0), d.angles.squeeze(0), d.atom_types, d.frac_coords)
        for d in batch.to_data_list()
    ]
    translated_batch = Batch.from_data_list(
        [
            Data(
                angles=torch.tensor(s.lattice.angles, dtype=torch.float32)[None],
                lengths=torch.tensor(s.lattice.lengths, dtype=torch.float32)[None],
                frac_coords=(torch.from_numpy(s.frac_coords).float() + torch.rand([1, 3])) % 1.0,
                atom_types=torch.tensor(s.atomic_numbers),
                num_atoms=s.num_sites,
                num_nodes=s.num_sites,
            )
            for s in structures
        ]
    )

    with torch.no_grad():
        out_normal_cells = model.forward(
            None,  # torch.zeros((batch.num_atoms.shape[0], 16)),
            batch.frac_coords,
            batch.atom_types,
            batch.num_atoms,
            batch.batch,
            batch.lengths,
            batch.angles,
        )

        out_translated = model.forward(
            None,  # torch.zeros((translated_batch.num_atoms.shape[0], 16)),
            translated_batch.frac_coords,
            translated_batch.atom_types,
            translated_batch.num_atoms,
            translated_batch.batch,
            translated_batch.lengths,
            translated_batch.angles,
        )

    torch.testing.assert_allclose(
        out_normal_cells.stress, out_translated.stress, atol=1e-4, rtol=1e-4
    )


def test_lattice_parameterization_invariance():
    """
    Tests whether our model's predicted score behaves as expected when choosing a different unit cell.
    """
    cutoff = 5.0
    max_neighbors = 200
    torch.manual_seed(2)
    model = get_model(
        max_neighbors=max_neighbors,
        cutoff=cutoff,
        # regress stress in a non-conservative way
        regress_stress=True,
        max_cell_images_per_dim=30,
    )
    model.eval()
    batch = get_mp_20_debug_batch()

    structures = [
        make_structure(d.lengths.squeeze(0), d.angles.squeeze(0), d.atom_types, d.frac_coords)
        for d in batch.to_data_list()
    ]
    lattice_matrices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
    lattice_matrix_changed = lattice_matrices.clone()

    # Build updated lattice matrices, where a random lattice vector is modified by adding 3x another random (different) lattice vector.
    # This modification does not change the underlying periodic structure.
    combs = torch.tensor(list(permutations(range(3), 2)))
    # Per lattice, select a random pair of lattice vectors, where we add 3x the second to the first one.
    lattice_vector_combine_ixs = torch.randint(0, len(combs), (lattice_matrices.shape[0],))
    combs_sel = combs[lattice_vector_combine_ixs]

    # Build the lattice perturbation matrices. For example, if the two lattice vectors are 0 and 1, we get the following:
    # [
    #   [1.0, 0.0, 0.0],
    #   [3.0, 1.0, 0.0],
    #   [0.0, 0.0, 1.0]
    # ],
    # which has the effect of changing the first lattice vector to be l_1 := l_1 + 3 * l_2 in the updated lattice.
    # Shape [batch_size, 3, 3]
    change_matrix = torch.eye(3)[None].expand_as(lattice_matrices).clone().contiguous()
    change_matrix[range(combs_sel.shape[0]), combs_sel[:, 0], combs_sel[:, 1]] = 3
    # Transposing is needed because in our model, the lattice is a stack of row lattice vectors, but the equations are for stacks of column matrices.
    lattice_matrix_changed = (lattice_matrices.transpose(1, 2) @ change_matrix).transpose(1, 2)
    new_frac_coords = cart_to_frac_coords_with_lattice(
        frac_to_cart_coords_with_lattice(batch.frac_coords, batch.num_atoms, lattice_matrices),
        batch.num_atoms,
        lattice_matrix_changed,
    )

    # Build new batch
    updated_batch = batch.clone()
    new_lengths, new_angles = lattice_matrix_to_params_torch(lattice_matrix_changed)
    updated_batch.frac_coords = new_frac_coords
    updated_batch.lengths = new_lengths
    updated_batch.angles = new_angles
    structures_perm = [
        make_structure(d.lengths.squeeze(0), d.angles.squeeze(0), d.atom_types, d.frac_coords)
        for d in updated_batch.to_data_list()
    ]

    # Make sure that pairwise distances haven't changed
    close = [
        torch.allclose(
            torch.from_numpy(structures_perm[ix].distance_matrix),
            torch.from_numpy(structures[ix].distance_matrix),
            atol=1e-3,
        )
        for ix in range(len(structures))
    ]
    assert all(close)

    # Forward the two batches
    with torch.no_grad():
        out_normal_cells = model.forward(
            None,
            batch.frac_coords,
            batch.atom_types,
            batch.num_atoms,
            batch.batch,
            lattice=lattice_matrices,
        )

        out_updated_batch = model.forward(
            None,
            updated_batch.frac_coords,
            updated_batch.atom_types,
            updated_batch.num_atoms,
            updated_batch.batch,
            lattice=lattice_matrix_changed,
        )
    assert not torch.allclose(
        change_matrix.inverse() @ out_normal_cells.stress, out_updated_batch.stress, atol=1e-3
    )
    assert not torch.allclose(out_normal_cells.stress, out_updated_batch.stress, atol=1e-3)


def test_symmetric_lattice_score():
    # test whether predicted stress via symmetric lattice updates is actually symmetric
    model = get_model(
        max_neighbors=20,
        cutoff=7.0,
        # regress stress in a non-conservative way
        regress_stress=True,
        max_cell_images_per_dim=20,
    )
    model.eval()
    batch = get_mp_20_debug_batch()

    with torch.no_grad():
        model_out = model.forward(
            None,
            batch.frac_coords,
            batch.atom_types,
            batch.num_atoms,
            batch.batch,
            batch.lengths,
            batch.angles,
        )

    # for mypy
    assert model_out.stress is not None
    assert torch.allclose(model_out.stress, model_out.stress.transpose(1, 2), atol=1e-5)


def test_rotation_invariance():
    model = get_model(
        max_neighbors=1000,
        cutoff=5.0,
        regress_stress=True,
        max_cell_images_per_dim=10,
    )

    batch = get_mp_20_debug_batch()
    lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
    with torch.no_grad():
        model_out = model.forward(
            None,
            batch.frac_coords,
            batch.atom_types,
            batch.num_atoms,
            batch.batch,
            lattice=lattices,
        )

    rotation_matrix = torch.Tensor(Rotation.random().as_matrix(), device=lattices.device)
    rotated_lattices = lattices @ rotation_matrix
    with torch.no_grad():
        model_out_rotated = model.forward(
            None,
            batch.frac_coords,
            batch.atom_types,
            batch.num_atoms,
            batch.batch,
            lattice=rotated_lattices,
        )

    forces = model_out.forces
    forces_rotated = model_out_rotated.forces
    stress = model_out.stress
    stress_rotated = model_out_rotated.stress

    assert torch.allclose(forces @ rotation_matrix, forces_rotated, atol=1e-3)

    assert torch.allclose(rotation_matrix.T @ stress @ rotation_matrix, stress_rotated, atol=1e-3)
