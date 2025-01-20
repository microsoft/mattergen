# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable

import torch

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.types import PropertySourceId
from mattergen.denoiser import GemNetTDenoiser, get_chemgraph_from_denoiser_output
from mattergen.property_embeddings import (
    ZerosEmbedding,
    get_property_embeddings,
    get_use_unconditional_embedding,
)

BatchTransform = Callable[[ChemGraph], ChemGraph]


class GemNetTAdapter(GemNetTDenoiser):
    """
    Denoiser layerwise adapter with GemNetT. On top of a mattergen.denoiser.GemNetTDenoiser,
    additionally inputs <property_embeddings_adapt> that specifies extra conditions to be conditioned on.
    """

    def __init__(self, property_embeddings_adapt: torch.nn.ModuleDict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ModuleDict[PropertyName, PropertyEmbedding] -- conditions adding by this adapter
        self.property_embeddings_adapt = torch.nn.ModuleDict(property_embeddings_adapt)

        # sanity check keys are required by the adapter that already exist in the base model
        assert all(
            [
                k not in self.property_embeddings.keys()
                for k in self.property_embeddings_adapt.keys()
            ]
        ), f"One of adapter conditions {self.property_embeddings_adapt.keys()} already exists in base model {self.property_embeddings.keys()}, please remove."

        # we make the choice that new adapter fields do not alter the unconditional score
        # we therefore need the unconditional embedding for all properties added in the adapter
        # to return 0. We hack the unconditional embedding module here to achieve that
        for property_embedding in self.property_embeddings_adapt.values():
            property_embedding.unconditional_embedding_module = ZerosEmbedding(
                hidden_dim=property_embedding.unconditional_embedding_module.hidden_dim,
            )

    def forward(
        self,
        x: ChemGraph,
        t: torch.Tensor,
    ) -> ChemGraph:
        """
        augment <z_per_crystal> with <self.condition_embs_adapt>.
        """
        (frac_coords, lattice, atom_types, num_atoms, batch,) = (
            x["pos"],
            x["cell"],
            x["atomic_numbers"],
            x["num_atoms"],
            x.get_batch_idx("pos"),
        )
        # (num_atoms, hidden_dim) (num_crysts, 3)
        t_enc = self.noise_level_encoding(t).to(lattice.device)
        z_per_crystal = t_enc

        # shape = (Nbatch, sum(hidden_dim of all properties in condition_on_adapt))
        conditions_base_model: torch.Tensor = get_property_embeddings(
            property_embeddings=self.property_embeddings, batch=x
        )

        if len(conditions_base_model) > 0:
            z_per_crystal = torch.cat([z_per_crystal, conditions_base_model], dim=-1)

        # compose into a dict
        conditions_adapt_dict = {}
        conditions_adapt_mask_dict = {}
        for cond_field, property_embedding in self.property_embeddings_adapt.items():
            conditions_adapt_dict[cond_field] = property_embedding.forward(batch=x)
            try:
                conditions_adapt_mask_dict[cond_field] = get_use_unconditional_embedding(
                    batch=x, cond_field=cond_field
                )
            except KeyError:
                # no values have been provided for the conditional field,
                # interpret this as the user wanting an unconditional score
                conditions_adapt_mask_dict[cond_field] = torch.ones_like(
                    x["num_atoms"], dtype=torch.bool
                ).reshape(-1, 1)

        output = self.gemnet(
            z=z_per_crystal,
            frac_coords=frac_coords,
            atom_types=atom_types,
            num_atoms=num_atoms,
            batch=batch,
            lengths=None,
            angles=None,
            lattice=lattice,
            # we construct the graph on the fly, hence pass None for these:
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
            cond_adapt=conditions_adapt_dict,
            cond_adapt_mask=conditions_adapt_mask_dict,  # when True use unconditional embedding
        )

        pred_atom_types = self.fc_atom(output.node_embeddings)

        return get_chemgraph_from_denoiser_output(
            pred_atom_types=pred_atom_types,
            pred_lattice_eps=output.stress,
            pred_cart_pos_eps=output.forces,
            training=self.training,
            element_mask_func=self.element_mask_func,
            x_input=x,
        )

    @property
    def cond_fields_model_was_trained_on(self) -> list[PropertySourceId]:
        """
        We adopt the convention that all property embeddings are stored in torch.nn.ModuleDicts of
        name property_embeddings or property_embeddings_adapt in the case of a fine tuned model.

        This function returns the list of all field names that a given score model was trained to
        condition on.
        """
        return list(self.property_embeddings) + list(self.property_embeddings_adapt)
