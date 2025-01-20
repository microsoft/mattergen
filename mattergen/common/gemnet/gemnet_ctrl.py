# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Adapted from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet/gemnet.py.

from typing import Dict, List, Optional

# import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter

from mattergen.common.data.types import PropertySourceId
from mattergen.common.gemnet.gemnet import GemNetT, ModelOutput
from mattergen.common.gemnet.utils import inner_product_normalized
from mattergen.common.utils.data_utils import (
    frac_to_cart_coords_with_lattice,
    lattice_params_to_matrix_torch,
)


class GemNetTCtrl(GemNetT):
    """
    GemNet-T, triplets-only variant of GemNet

    This variation allows for layerwise conditional control for the purpose of
    conditional finetuning. It adds the following on top of GemNetT:

    for each condition in <condition_on_adapt>:

    1. a series of adapt layers that take the concatenation of the node embedding
       and the condition embedding, process it with an MLP. There is one adapt layer
       for each GemNetT message passing block.
    2. a series of mixin layers that take the output of the adapt layer and mix it in
       to the atom embedding. There is one mixin layer for each GemNetT message passing block.
       The mixin layers are initialized to zeros so at the beginning of training, the model
       outputs exactly the same scores as the base GemNetT model.

    """

    def __init__(self, condition_on_adapt: List[PropertySourceId], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.condition_on_adapt = condition_on_adapt
        self.cond_adapt_layers = nn.ModuleDict()
        self.cond_mixin_layers = nn.ModuleDict()
        # default value for emb_size_atom is 512
        self.emb_size_atom = kwargs["emb_size_atom"] if "emb_size_atom" in kwargs else 512

        for cond in condition_on_adapt:
            adapt_layers = []
            mixin_layers = []

            for _ in range(self.num_blocks):
                adapt_layers.append(
                    nn.Sequential(
                        nn.Linear(self.emb_size_atom * 2, self.emb_size_atom),
                        nn.ReLU(),
                        nn.Linear(self.emb_size_atom, self.emb_size_atom),
                    )
                )
                mixin_layers.append(nn.Linear(self.emb_size_atom, self.emb_size_atom, bias=False))
                nn.init.zeros_(mixin_layers[-1].weight)

            self.cond_adapt_layers[cond] = torch.nn.ModuleList(adapt_layers)
            self.cond_mixin_layers[cond] = torch.nn.ModuleList(mixin_layers)

    def forward(
        self,
        z: torch.Tensor,
        frac_coords: torch.Tensor,
        atom_types: torch.Tensor,
        num_atoms: torch.Tensor,
        batch: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        angles: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        to_jimages: Optional[torch.Tensor] = None,
        num_bonds: Optional[torch.Tensor] = None,
        lattice: Optional[torch.Tensor] = None,
        charges: Optional[torch.Tensor] = None,
        cond_adapt: Optional[Dict[PropertySourceId, torch.Tensor]] = None,
        cond_adapt_mask: Optional[Dict[PropertySourceId, torch.Tensor]] = None,
    ) -> ModelOutput:
        """
        args:
            z: (N_cryst, num_latent)
            frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, ) with D3PM need to use atomic number
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3) (optional, either lengths and angles or lattice must be passed)
            angles: (N_cryst, 3) (optional, either lengths and angles or lattice must be passed)
            edge_index: (2, N_edge) (optional, only needed if self.otf_graph is False)
            to_jimages: (N_edge, 3) (optional, only needed if self.otf_graph is False)
            num_bonds: (N_cryst,) (optional, only needed if self.otf_graph is False)
            lattice: (N_cryst, 3, 3) (optional, either lengths and angles or lattice must be passed)
            cond_adapt: (N_cryst, num_cond, dim_cond) (optional, conditional signal for score prediction)
            cond_adapt_mask: (N_cryst, num_cond) (optional, mask for which data points receive conditional signal)
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
        # Merge z and atom embedding
        if z is not None:
            z_per_atom = z[batch]
            h = torch.cat([h, z_per_atom], dim=1)
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

        # currently only working for a single cond adapt property.
        # to extend to multi-properties,
        # use a ModuleDict for adapt layers and mixin layers.
        # use a dictionary to track the conditions?

        if cond_adapt is not None and cond_adapt_mask is not None:
            cond_adapt_per_atom = {}
            cond_adapt_mask_per_atom = {}
            for cond in self.condition_on_adapt:
                cond_adapt_per_atom[cond] = cond_adapt[cond][batch]
                # 1 = use conditional embedding, 0 = use unconditional embedding
                cond_adapt_mask_per_atom[cond] = 1.0 - cond_adapt_mask[cond][batch].float()

        for i in range(self.num_blocks):
            h_adapt = torch.zeros_like(h)
            for cond in self.condition_on_adapt:
                h_adapt_cond = self.cond_adapt_layers[cond][i](
                    torch.cat([h, cond_adapt_per_atom[cond]], dim=-1)
                )
                h_adapt_cond = self.cond_mixin_layers[cond][i](h_adapt_cond)
                # cond_adapt_mask_per_atom[cond] is 1.0 if we want to use conditional embedding and 0 for unconditional embedding
                h_adapt += cond_adapt_mask_per_atom[cond] * h_adapt_cond
            h = h + h_adapt

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
        output["forces"] = F_t

        if self.regress_stress:
            # shape=(Nbatch, 3, 3)
            output["stress"] = lattice_update

        return ModelOutput(**output)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
