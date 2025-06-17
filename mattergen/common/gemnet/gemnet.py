"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
Adapted from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet/gemnet.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

# import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_sparse import SparseTensor

from mattergen.common.gemnet.layers.atom_update_block import OutputBlock
from mattergen.common.gemnet.layers.base_layers import Dense
from mattergen.common.gemnet.layers.efficient import EfficientInteractionDownProjection
from mattergen.common.gemnet.layers.embedding_block import EdgeEmbedding
from mattergen.common.gemnet.layers.interaction_block import InteractionBlockTripletsOnly
from mattergen.common.gemnet.layers.radial_basis import RadialBasis
from mattergen.common.gemnet.layers.scaling import AutomaticFit
from mattergen.common.gemnet.layers.spherical_basis import CircularBasisLayer
from mattergen.common.gemnet.utils import (
    inner_product_normalized,
    mask_neighbors,
    ragged_range,
    repeat_blocks,
)
from mattergen.common.utils.data_utils import (
    frac_to_cart_coords_with_lattice,
    get_pbc_distances,
    lattice_params_to_matrix_torch,
    radius_graph_pbc,
)
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT, get_device, get_pyg_device
from mattergen.common.utils.lattice_score import edge_score_to_lattice_score_frac_symmetric


@dataclass(frozen=True)
class ModelOutput:
    energy: torch.Tensor
    node_embeddings: torch.Tensor
    forces: Optional[torch.Tensor] = None
    stress: Optional[torch.Tensor] = None


class RBFBasedLatticeUpdateBlock(torch.nn.Module):
    # Lattice update block that mimics GemNet's edge processing, e.g., uses radial basis functions.
    def __init__(
        self,
        emb_size: int,
        activation: str,
        emb_size_rbf: int,
        emb_size_edge: int,
        num_heads: int = 1,
    ):
        super().__init__()
        self.num_out = num_heads
        self.mlp = nn.Sequential(
            Dense(emb_size, emb_size, activation=activation), Dense(emb_size, emb_size)
        )
        self.dense_rbf_F = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.out_forces = Dense(emb_size_edge, num_heads, bias=False, activation=None)

    def compute_score_per_edge(
        self,
        edge_emb: torch.Tensor,  # [Num_edges, emb_dim]
        rbf: torch.Tensor,  # [Num_edges, num_rbf_bases]
    ) -> torch.Tensor:
        x_F = self.mlp(edge_emb)
        rbf_emb_F = self.dense_rbf_F(rbf)  # (nEdges, emb_size_edge)
        x_F_rbf = x_F * rbf_emb_F
        # x_F = self.scale_rbf_F(x_F, x_F_rbf)
        x_F = self.out_forces(x_F_rbf)  # (nEdges, self.num_out)
        return x_F


class RBFBasedLatticeUpdateBlockFrac(RBFBasedLatticeUpdateBlock):
    # Lattice update block that mimics GemNet's edge processing, e.g., uses radial basis functions.
    def __init__(
        self,
        emb_size: int,
        activation: str,
        emb_size_rbf: int,
        emb_size_edge: int,
        num_heads: int = 1,
    ):
        super().__init__(
            emb_size=emb_size,
            activation=activation,
            emb_size_rbf=emb_size_rbf,
            emb_size_edge=emb_size_edge,
            num_heads=num_heads,
        )

    def forward(
        self,
        edge_emb: torch.Tensor,  # [Num_edges, emb_dim]
        edge_index: torch.Tensor,  # [2, Num_edges]
        distance_vec: torch.Tensor,  # [Num_edges, 3]
        lattice: torch.Tensor,  # [Num_crystals, 3, 3]
        batch: torch.Tensor,  # [Num_atoms, ]
        rbf: torch.Tensor,  # [Num_edges, num_rbf_bases]
        normalize_score: bool = True,
    ) -> torch.Tensor:
        edge_scores = self.compute_score_per_edge(edge_emb=edge_emb, rbf=rbf)
        if normalize_score:
            num_edges = scatter(torch.ones_like(distance_vec[:, 0]), batch[edge_index[0]])
            edge_scores /= num_edges[batch[edge_index[0]], None]
        outs = []
        for i in range(self.num_out):
            lattice_update = edge_score_to_lattice_score_frac_symmetric(
                score_d=edge_scores[:, i],
                edge_index=edge_index,
                edge_vectors=distance_vec,
                batch=batch,
            )
            outs.append(lattice_update)
        outs = torch.stack(outs, dim=-1).sum(-1)
        # [Batch_size, 3, 3]
        return outs


class GemNetT(torch.nn.Module):
    """
    GemNet-T, triplets-only variant of GemNet

    Parameters
    ----------
        num_targets: int
            Number of prediction targets.

        num_spherical: int
            Controls maximum frequency.
        num_radial: int
            Controls maximum frequency.
        num_blocks: int
            Number of building blocks to be stacked.

        atom_embedding: torch.nn.Module
            a module that embeds atomic numbers into vectors of size emb_dim_atomic_number.
        emb_size_atom: int
            Embedding size of the atoms. This can be different from emb_dim_atomic_number.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.
        cutoff: float
            Embedding cutoff for interactomic directions in Angstrom.
        rbf: dict
            Name and hyperparameters of the radial basis function.
        envelope: dict
            Name and hyperparameters of the envelope function.
        cbf: dict
            Name and hyperparameters of the cosine basis function.
        output_init: str
            Initialization method for the final dense layer.
        activation: str
            Name of the activation function.
        scale_file: str
            Path to the json file containing the scaling factors.
        encoder_mode: bool
            if <True>, use the encoder mode of the model, i.e. only get the atom/edge embedddings.
    """

    def __init__(
        self,
        num_targets: int,
        latent_dim: int,
        atom_embedding: torch.nn.Module,
        atom_additional_property_dim: int = 0,
        num_spherical: int = 7,
        num_radial: int = 128,
        num_blocks: int = 3,
        emb_size_atom: int = 512,
        emb_size_edge: int = 512,
        emb_size_trip: int = 64,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        emb_size_bil_trip: int = 64,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_concat: int = 1,
        num_atom: int = 3,
        regress_stress: bool = False,
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        otf_graph: bool = False,
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        max_cell_images_per_dim: int = 5,
        encoder_mode: bool = False,  #
        **kwargs,
    ):
        super().__init__()
        scale_file = f"{MODELS_PROJECT_ROOT}/common/gemnet/gemnet-dT.json"
        assert scale_file is not None, "`scale_file` is required."

        self.encoder_mode = encoder_mode
        self.num_targets = num_targets
        assert num_blocks > 0
        self.num_blocks = num_blocks
        emb_dim_atomic_number = getattr(atom_embedding, "emb_size")

        self.cutoff = cutoff

        self.max_neighbors = max_neighbors

        self.max_cell_images_per_dim = max_cell_images_per_dim

        self.otf_graph = otf_graph

        self.regress_stress = regress_stress
        # we might want to take care of permutation invariance w.r.t. the order of the lattice vectors, though I don't think this is critical.
        self.angle_edge_emb = nn.Sequential(
            nn.Linear(emb_size_edge + 3, emb_size_edge),
            nn.ReLU(),
            nn.Linear(emb_size_edge, emb_size_edge),
        )

        AutomaticFit.reset()  # make sure that queue is empty (avoid potential error)

        # ---------------------------------- Basis Functions ---------------------------------- ###
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        radial_basis_cbf3 = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )
        self.cbf_basis3 = CircularBasisLayer(
            num_spherical,
            radial_basis=radial_basis_cbf3,
            cbf=cbf,
            efficient=True,
        )
        # ------------------------------------------------------------------------------------- ###

        # --------------------------------- Update lattice MLP -------------------------------- ###
        self.regress_stress = regress_stress
        self.lattice_out_blocks = nn.ModuleList(
            [
                RBFBasedLatticeUpdateBlockFrac(
                    emb_size_edge,
                    activation,
                    emb_size_rbf,
                    emb_size_edge,
                )
                for _ in range(num_blocks + 1)
            ]
        )
        self.mlp_rbf_lattice = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        # ------------------------------------------------------------------------------------- ###
        # ------------------------------- Share Down Projections ------------------------------ ###
        # Share down projection across all interaction blocks
        self.mlp_rbf3 = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(num_spherical, num_radial, emb_size_cbf)

        # Share the dense Layer of the atom embedding block across the interaction blocks
        self.mlp_rbf_h = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        # ------------------------------------------------------------------------------------- ###

        self.atom_emb = atom_embedding
        self.atom_additional_property_dim = atom_additional_property_dim
        self.atom_latent_emb = nn.Linear(emb_dim_atomic_number + latent_dim + atom_additional_property_dim, emb_size_atom)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )

        out_blocks = []
        int_blocks = []

        # Interaction Blocks
        interaction_block = InteractionBlockTripletsOnly  # GemNet-(d)T
        for i in range(num_blocks):
            int_blocks.append(
                interaction_block(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_bil_trip=emb_size_bil_trip,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    activation=activation,
                    scale_file=scale_file,
                    name=f"IntBlock_{i+1}",
                )
            )

        for i in range(num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    num_targets=num_targets,
                    activation=activation,
                    output_init=output_init,
                    direct_forces=True,
                    scale_file=scale_file,
                    name=f"OutBlock_{i}",
                )
            )

        self.out_blocks = torch.nn.ModuleList(out_blocks)
        self.int_blocks = torch.nn.ModuleList(int_blocks)

        self.shared_parameters = [
            (self.mlp_rbf3, self.num_blocks),
            (self.mlp_cbf3, self.num_blocks),
            (self.mlp_rbf_h, self.num_blocks),
            (self.mlp_rbf_out, self.num_blocks + 1),
        ]

    def get_triplets(
        self, edge_index: torch.Tensor, num_atoms: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all b->a for each edge c->a.
        It is possible that b=c, as long as the edges are distinct.

        Returns
        -------
        id3_ba: torch.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        id3_ca: torch.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        id3_ragged_idx: torch.Tensor, shape (num_triplets,)
            Indices enumerating the copies of id3_ca for creating a padded matrix
        """
        idx_s, idx_t = edge_index  # c->a (source=c, target=a)

        value = torch.arange(idx_s.size(0), device=idx_s.device, dtype=idx_s.dtype)
        # Possibly contains multiple copies of the same edge (for periodic interactions)
        pyg_device = get_pyg_device() if idx_s.device != torch.device("cpu") else idx_s.device
        torch_device = get_device() if idx_s.device != torch.device("cpu") else idx_s.device
        adj = SparseTensor(
            row=idx_t.to(pyg_device),
            col=idx_s.to(pyg_device),
            value=value.to(pyg_device),
            sparse_sizes=(num_atoms.to(pyg_device), num_atoms.to(pyg_device)),
        )
        adj_edges = adj[idx_t.to(pyg_device)].to(torch_device)

        # Edge indices (b->a, c->a) for triplets.
        id3_ba = adj_edges.storage.value().to(torch_device)
        id3_ca = adj_edges.storage.row().to(torch_device)

        # Remove self-loop triplets
        # Compare edge indices, not atom indices to correctly handle periodic interactions
        mask = id3_ba != id3_ca
        id3_ba = id3_ba[mask]
        id3_ca = id3_ca[mask]

        # Get indices to reshape the neighbor indices b->a into a dense matrix.
        # id3_ca has to be sorted for this to work.
        num_triplets = torch.bincount(id3_ca, minlength=idx_s.size(0))
        id3_ragged_idx = ragged_range(num_triplets)

        return id3_ba, id3_ca, id3_ragged_idx

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
        self,
        edge_index: torch.Tensor,
        cell_offsets: torch.Tensor,
        neighbors: torch.Tensor,
        edge_dist: torch.Tensor,
        edge_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] == 0) & (cell_offsets[:, 2] < 0))
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(batch_edge, minlength=neighbors.size(0))

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(cell_offsets, mask, edge_reorder_idx, True)
        edge_dist_new = self.select_symmetric_edges(edge_dist, mask, edge_reorder_idx, False)
        edge_vector_new = self.select_symmetric_edges(edge_vector, mask, edge_reorder_idx, True)

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_dist_new,
            edge_vector_new,
        )

    def select_edges(
        self,
        edge_index: torch.Tensor,
        cell_offsets: torch.Tensor,
        neighbors: torch.Tensor,
        edge_dist: torch.Tensor,
        edge_vector: torch.Tensor,
        cutoff: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if cutoff is not None:
            edge_mask = edge_dist <= cutoff

            edge_index = edge_index[:, edge_mask]
            cell_offsets = cell_offsets[edge_mask]
            neighbors = mask_neighbors(neighbors, edge_mask)
            edge_dist = edge_dist[edge_mask]
            edge_vector = edge_vector[edge_mask]

        return edge_index, cell_offsets, neighbors, edge_dist, edge_vector

    def generate_interaction_graph(
        self,
        cart_coords: torch.Tensor,
        lattice: torch.Tensor,
        num_atoms: torch.Tensor,
        edge_index: torch.Tensor,
        to_jimages: torch.Tensor,
        num_bonds: torch.Tensor,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.otf_graph:
            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords=cart_coords,
                lattice=lattice,
                num_atoms=num_atoms,
                radius=self.cutoff,
                max_num_neighbors_threshold=self.max_neighbors,
                max_cell_images_per_dim=self.max_cell_images_per_dim,
            )

        # Switch the indices, so the second one becomes the target index,
        # over which we can efficiently aggregate.
        out = get_pbc_distances(
            cart_coords,
            edge_index,
            lattice,
            to_jimages,
            num_atoms,
            num_bonds,
            coord_is_cart=True,
            return_offsets=True,
            return_distance_vec=True,
        )

        edge_index = out["edge_index"]
        D_st = out["distances"]
        # These vectors actually point in the opposite direction.
        # But we want to use col as idx_t for efficient aggregation.
        V_st = -out["distance_vec"] / D_st[:, None]

        (
            edge_index,
            cell_offsets,
            neighbors,
            D_st,
            V_st,
        ) = self.reorder_symmetric_edges(edge_index, to_jimages, num_bonds, D_st, V_st)

        # Indices for swapping c->a and a->c (for symmetric MP)
        block_sizes = neighbors // 2

        # Remove 0 sizes
        block_sizes = torch.masked_select(block_sizes, block_sizes > 0)
        id_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )

        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(
            edge_index,
            num_atoms=num_atoms.sum(),
        )

        return (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            cell_offsets,
        )

    def forward(
        self,
        z: torch.Tensor,
        frac_coords: torch.Tensor,
        atom_types: torch.Tensor,
        intercalant_origin: torch.Tensor,
        num_atoms: torch.Tensor,
        batch: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        angles: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        to_jimages: Optional[torch.Tensor] = None,
        num_bonds: Optional[torch.Tensor] = None,
        lattice: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        args:
            z: (N_cryst, num_latent)
            frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, ) with D3PM need to use atomic number
            intercalant_origin: (N_atoms, )
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3) (optional, either lengths and angles or lattice must be passed)
            angles: (N_cryst, 3) (optional, either lengths and angles or lattice must be passed)
            edge_index: (2, N_edge) (optional, only needed if self.otf_graph is False)
            to_jimages: (N_edge, 3) (optional, only needed if self.otf_graph is False)
            num_bonds: (N_cryst,) (optional, only needed if self.otf_graph is False)
            lattice: (N_cryst, 3, 3) (optional, either lengths and angles or lattice must be passed)
        returns:
            atom_frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """

        if self.otf_graph:
            assert all(
                [edge_index is None, to_jimages is None, num_bonds is None]
            ), "OTF graph construction is active but received input graph information."
        else:
            assert not any(
                [edge_index is None, to_jimages is None, num_bonds is None]
            ), "OTF graph construction is off but received no input graph information."

        assert (angles is None and lengths is None) != (
            lattice is None
        ), "Either lattice or lengths and angles must be provided, not both or none."
        if angles is not None and lengths is not None:
            lattice = lattice_params_to_matrix_torch(lengths, angles)
        assert lattice is not None
        distorted_lattice = lattice

        pos = frac_to_cart_coords_with_lattice(frac_coords, num_atoms, lattice=distorted_lattice)

        atomic_numbers = atom_types

        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            to_jimages,
        ) = self.generate_interaction_graph(
            pos, distorted_lattice, num_atoms, edge_index, to_jimages, num_bonds
        )
        idx_s, idx_t = edge_index

        # Calculate triplet angles
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)

        rbf = self.radial_basis(D_st)

        # Embedding block
        h = self.atom_emb(atomic_numbers)
        if intercalant_origin is not None:
            h = torch.cat([h, intercalant_origin.float().unsqueeze(-1)], dim=1)
        # Merge z and atom embedding
        if z is not None:
            z_per_atom = z[batch]
            h = torch.cat([h, z_per_atom], dim=1)
            # Combine all embeddings
            h = self.atom_latent_emb(h)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, rbf, idx_s, idx_t)  # (nEdges, emb_size_edge)
        batch_edge = batch[edge_index[0]]
        cosines = torch.cosine_similarity(V_st[:, None], distorted_lattice[batch_edge], dim=-1)
        m = torch.cat([m, cosines], dim=-1)
        m = self.angle_edge_emb(m)

        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)

        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)

        E_t, F_st = self.out_blocks[0](h, m, rbf_out, idx_t)

        distance_vec = V_st * D_st[:, None]

        lattice_update = None
        rbf_lattice = self.mlp_rbf_lattice(rbf)
        lattice_update = self.lattice_out_blocks[0](
            edge_emb=m,
            edge_index=edge_index,
            distance_vec=distance_vec,
            lattice=distorted_lattice,
            batch=batch,
            rbf=rbf_lattice,
            normalize_score=True,
        )
        F_fully_connected = torch.tensor(0.0, device=distorted_lattice.device)
        for i in range(self.num_blocks):
            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id3_ragged_idx=id3_ragged_idx,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            E, F = self.out_blocks[i + 1](h, m, rbf_out, idx_t)
            # (nAtoms, num_targets), (nEdges, num_targets)
            F_st += F
            E_t += E
            rbf_lattice = self.mlp_rbf_lattice(rbf)
            lattice_update += self.lattice_out_blocks[i + 1](
                edge_emb=m,
                edge_index=edge_index,
                distance_vec=distance_vec,
                lattice=distorted_lattice,
                batch=batch,
                rbf=rbf_lattice,
                normalize_score=True,
            )

        nMolecules = torch.max(batch) + 1

        if self.encoder_mode:
            return E_t
        # always use sum aggregation
        E_t = scatter(
            E_t, batch, dim=0, dim_size=nMolecules, reduce="sum"
        )  # (nMolecules, num_targets)

        # always output energy, forces and node embeddings
        output = dict(energy=E_t, node_embeddings=h)

        # map forces in edge directions
        F_st_vec = F_st[:, :, None] * V_st[:, None, :]
        # (nEdges, num_targets, 3)
        F_t = scatter(
            F_st_vec,
            idx_t,
            dim=0,
            dim_size=num_atoms.sum(),
            reduce="add",
        )  # (nAtoms, num_targets, 3)
        F_t = F_t.squeeze(1)  # (nAtoms, 3)
        output["forces"] = F_t + F_fully_connected

        if self.regress_stress:
            # optionally get predicted stress tensor
            # shape=(Nbatch, 3, 3)
            output["stress"] = lattice_update

        return ModelOutput(**output)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
