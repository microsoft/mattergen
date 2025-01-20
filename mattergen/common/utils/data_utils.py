# Copyright (c) 2021 Tian Xie, Xiang Fu
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Adapted from https://github.com/txie-93/cdvae/blob/main/cdvae/common/data_utils.py

from functools import lru_cache

import numpy as np
import torch
from pymatgen.core import Element

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.utils.ocp_graph_utils import radius_graph_pbc as radius_graph_pbc_ocp

EPSILON = 1e-5


@lru_cache
def get_atomic_number(symbol: str) -> int:
    # get atomic number from Element symbol
    return Element(symbol).Z


@lru_cache
def get_element_symbol(Z: int) -> str:
    # get Element symbol from atomic number
    return str(Element.from_Z(Z=Z))


def abs_cap(val: float, max_abs_val: float = 1.0) -> float:
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trigonometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def lattice_params_to_matrix_torch(
    lengths: torch.Tensor, angles: torch.Tensor, eps: float = 0.0
) -> torch.Tensor:
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    coses = torch.clamp(torch.cos(torch.deg2rad(angles)), -1.0, 1.0)
    sins = (1 - coses**2).sqrt()

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = torch.clamp(val, -1.0 + eps, 1.0 - eps)

    vector_a = torch.stack(
        [
            lengths[:, 0] * sins[:, 1],
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 0] * coses[:, 1],
        ],
        dim=1,
    )
    vector_b = torch.stack(
        [
            -lengths[:, 1] * sins[:, 0] * val,
            lengths[:, 1] * sins[:, 0] * (1 - val**2).sqrt(),
            lengths[:, 1] * coses[:, 0],
        ],
        dim=1,
    )
    vector_c = torch.stack(
        [
            torch.zeros(lengths.size(0), device=lengths.device),
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 2],
        ],
        dim=1,
    )

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def lattice_matrix_to_params_torch(
    matrix: torch.Tensor, eps: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a batch of lattice matrices into their corresponding unit cell vector lengths and angles.

    Args:
        matrix (torch.Tensor, [B, 3, 3]): The batch of lattice matrices.

    Returns:
        tuple[torch.Tensor], ([B, 3], [B, 3]): tuple whose first element is the lengths of the unit cell vectors, and the second one gives the angles between the vectors.
    """
    assert len(matrix.shape) == 3

    # derivatives of arccos(cos(theta)) are undefined for abs(cos(theta))=1
    # we should physically encounter lattices that have vectors that are
    # parallel to one another. NOTE: the value of eps may need tuning
    # if calculations are found to fail, reduce this magnitude

    lengths = matrix.norm(p=2, dim=-1)
    ix_j = torch.tensor([1, 2, 0], dtype=torch.long, device=matrix.device)
    ix_k = torch.tensor([2, 0, 1], dtype=torch.long, device=matrix.device)
    cos_angles = (torch.cosine_similarity(matrix[:, ix_j], matrix[:, ix_k], dim=-1)).clamp(
        -1 + eps, 1 - eps
    )
    if len(matrix.shape) == 2:
        cos_angles = cos_angles.squeeze(0)
        lengths = lengths.squeeze(0)
    return lengths, torch.arccos(cos_angles) * 180.0 / np.pi


def lattice_matrix_to_params(matrix: np.ndarray) -> tuple[float, float, float, float, float, float]:
    lengths = np.sqrt(np.sum(matrix**2, axis=1)).tolist()

    angles = np.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = abs_cap(np.dot(matrix[j], matrix[k]) / (lengths[j] * lengths[k]))
    angles = np.arccos(angles) * 180.0 / np.pi
    a, b, c = lengths
    alpha, beta, gamma = angles
    return a, b, c, alpha, beta, gamma


def frac_to_cart_coords(
    frac_coords: torch.Tensor, lengths: torch.Tensor, angles: torch.Tensor, num_atoms: torch.Tensor
) -> torch.Tensor:
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    return frac_to_cart_coords_with_lattice(frac_coords, num_atoms, lattice)


def cart_to_frac_coords(
    cart_coords: torch.Tensor, lengths: torch.Tensor, angles: torch.Tensor, num_atoms: torch.Tensor
) -> torch.Tensor:
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    return cart_to_frac_coords_with_lattice(cart_coords, num_atoms, lattice)


def frac_to_cart_coords_with_lattice(
    frac_coords: torch.Tensor, num_atoms: torch.Tensor, lattice: torch.Tensor
) -> torch.Tensor:
    lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
    pos = torch.einsum("bi,bij->bj", frac_coords, lattice_nodes)  # cart coords
    return pos


def cart_to_frac_coords_with_lattice(
    cart_coords: torch.Tensor, num_atoms: torch.Tensor, lattice: torch.Tensor
) -> torch.Tensor:
    # use pinv in case the predicted lattice is not rank 3
    inv_lattice = torch.linalg.pinv(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum("bi,bij->bj", cart_coords, inv_lattice_nodes)
    return frac_coords % 1.0


def get_pbc_distances(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    lattice: torch.Tensor,
    to_jimages: torch.Tensor,
    num_atoms: torch.Tensor,
    num_bonds: torch.Tensor,
    coord_is_cart: bool = False,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
) -> torch.Tensor:
    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
        pos = torch.einsum("bi,bij->bj", coords, lattice_nodes)  # cart coords

    j_index, i_index = edge_index

    distance_vectors = pos[j_index] - pos[i_index]

    # correct for pbc
    lattice_edges = torch.repeat_interleave(lattice, num_bonds, dim=0)
    offsets = torch.einsum("bi,bij->bj", to_jimages.float(), lattice_edges)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors

    if return_offsets:
        out["offsets"] = offsets

    return out


def radius_graph_pbc(
    cart_coords: torch.Tensor,
    lattice: torch.Tensor,
    num_atoms: torch.Tensor,
    radius: float,
    max_num_neighbors_threshold: int,
    max_cell_images_per_dim: int = 10,
    topk_per_pair: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes pbc graph edges under pbc.

    topk_per_pair: (num_atom_pairs,), select topk edges per atom pair

    Note: topk should take into account self-self edge for (i, i)

        Keyword arguments
        -----------------
        cart_cords.shape=[Ntotal, 3] -- concatenate all atoms over all crystals
        lattice.shape=[Ncrystal, 3, 3]
        num_atoms.shape=[Ncrystal]
        max_cell_images_per_dim -- constrain the max. number of cell images per dimension in event
                                that infinitesimal angles between lattice vectors are encountered.

    WARNING: It is possible (and has been observed) that for rare cases when periodic atom images are
    on or close to the cut off radius boundary, doing these operations in 32 bit floating point can
    lead to atoms being spuriously considered within or outside of the cut off radius. This can lead
    to invariance of the neighbour list under global translation of all atoms in the unit cell. For
    the rare cases where this was observed, switching to 64 bit precision solved the issue. Since all
    graph embeddings should taper messages from neighbours to zero at the cut off radius, the effect
    of these errors in 32-bit should be negligible in practice.
    """
    assert topk_per_pair is None, "non None values of topk_per_pair is not supported"
    edge_index, unit_cell, num_neighbors_image, _, _ = radius_graph_pbc_ocp(
        pos=cart_coords,
        cell=lattice,
        natoms=num_atoms,
        pbc=torch.Tensor([True, True, True])
        .to(torch.bool)
        .to(cart_coords.device),  # torch.BoolTensor([...],device='cuda') fails
        radius=radius,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        max_cell_images_per_dim=max_cell_images_per_dim,
    )
    return edge_index, unit_cell, num_neighbors_image


class StandardScalerTorch(torch.nn.Module):
    """Normalizes the targets of a dataset."""

    def __init__(
        self,
        means: torch.Tensor | None = None,
        stds: torch.Tensor | None = None,
        stats_dim: tuple[int] = (
            1,
        ),  # dimension of mean, std stats (= X.shape[1:] for some input tensor X)
    ):
        super().__init__()
        # we need to make sure that we initialize means and stds with the right shapes
        # otherwise, we cannot load checkpoints of fitted means/stds.
        # ignore stats_dim if means and stds are provided
        self.register_buffer(
            "means", torch.atleast_1d(means) if means is not None else torch.empty(stats_dim)
        )
        self.register_buffer(
            "stds", torch.atleast_1d(stds) if stds is not None else torch.empty(stats_dim)
        )

    @property
    def device(self) -> torch.device:
        return self.means.device  # type: ignore

    def fit(self, X: torch.Tensor):
        means: torch.Tensor = torch.atleast_1d(torch.nanmean(X, dim=0).to(self.device))
        stds: torch.Tensor = torch.atleast_1d(
            torch_nanstd(X, dim=0, unbiased=False).to(self.device) + EPSILON
        )
        # mypy gets really confused about variables registered via register_buffer,
        # so we need to ignore a lot of type errors below
        assert (
            means.shape == self.means.shape  # type: ignore
        ), f"Mean shape mismatch: {means.shape} != {self.means.shape}"  # type: ignore
        assert (
            stds.shape == self.stds.shape  # type: ignore
        ), f"Std shape mismatch: {stds.shape} != {self.stds.shape}"  # type: ignore
        self.means = means  # type: ignore
        self.stds = stds  # type: ignore

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        assert self.means is not None and self.stds is not None
        return (X - self.means) / self.stds

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        assert self.means is not None and self.stds is not None
        return X * self.stds + self.means

    def match_device(self, X: torch.Tensor) -> torch.Tensor:
        assert self.means.numel() > 0 and self.stds.numel() > 0
        if self.means.device != X.device:
            self.means = self.means.to(X.device)
            self.stds = self.stds.to(X.device)

    def copy(self) -> "StandardScalerTorch":
        return StandardScalerTorch(
            means=self.means.clone().detach(),
            stds=self.stds.clone().detach(),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.transform(X)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist() if self.means is not None else None}, "
            f"stds: {self.stds.tolist() if self.stds is not None else None})"
        )


def torch_nanstd(x: torch.Tensor, dim: int, unbiased: bool) -> torch.Tensor:
    data_is_present = torch.all(
        torch.reshape(torch.logical_not(torch.isnan(x)), (x.shape[0], -1)),
        dim=1,
    )
    # https://github.com/pytorch/pytorch/issues/29372
    return torch.std(x[data_is_present], dim=dim, unbiased=unbiased)


def compute_lattice_polar_decomposition(lattice_matrix: torch.Tensor) -> torch.Tensor:
    # Polar decomposition via SVD, see https://en.wikipedia.org/wiki/Polar_decomposition
    # lattice_matrix: [batch_size, 3, 3]
    # Computes the (unique) symmetric lattice matrix that is equivalent (up to rotation) to the input lattice.

    if lattice_matrix.device.type == "cuda":
        # there is an issue running torch.linalg.svd on cuda tensors with driver version 450.*

        try:
            W, S, V_transp = torch.linalg.svd(lattice_matrix)
        except torch._C._LinAlgError:
            # move to cpu and try again
            W, S, V_transp = torch.linalg.svd(lattice_matrix.to("cpu"))
            W = W.to(lattice_matrix.device.type)
            S = S.to(lattice_matrix.device.type)
            V_transp = V_transp.to(lattice_matrix.device.type)
    else:
        W, S, V_transp = torch.linalg.svd(lattice_matrix)
    S_square = torch.diag_embed(S)
    V = V_transp.transpose(1, 2)
    U = W @ V_transp
    P = V @ S_square @ V_transp
    P_prime = U @ P @ U.transpose(1, 2)
    # symmetrized lattice matrix
    symm_lattice_matrix = P_prime
    return symm_lattice_matrix


def create_chem_graph_from_composition(target_composition_dict: dict[str, float]) -> ChemGraph:
    atomic_numbers = []
    for element_name, number_of_atoms in target_composition_dict.items():
        atomic_numbers += [Element(element_name).Z] * int(number_of_atoms)

    return ChemGraph(
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        num_atoms=torch.tensor([len(atomic_numbers)], dtype=torch.long),
        cell=torch.eye(3, dtype=torch.float).reshape(1, 3, 3),
        pos=torch.zeros((len(atomic_numbers), 3), dtype=torch.float),
    )
