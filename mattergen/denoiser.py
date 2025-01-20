# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable

import torch
import torch.nn as nn

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.types import PropertySourceId
from mattergen.common.utils.globals import MAX_ATOMIC_NUM, SELECTED_ATOMIC_NUMBERS
from mattergen.diffusion.model_utils import NoiseLevelEncoding
from mattergen.diffusion.score_models.base import ScoreModel
from mattergen.property_embeddings import (
    ChemicalSystemMultiHotEmbedding,
    get_property_embeddings,
    get_use_unconditional_embedding,
)

BatchTransform = Callable[[ChemGraph], ChemGraph]


def atomic_numbers_to_mask(atomic_numbers: torch.LongTensor, max_atomic_num: int) -> torch.Tensor:
    """Convert atomic numbers to a mask.

    Args:
        atomic_numbers (torch.LongTensor): One-based atomic numbers of shape (batch_size, )

    Returns:
        torch.Tensor: Mask of shape (batch_size, num_classes)
    """
    k_hot_mask = torch.eye(max_atomic_num, device=atomic_numbers.device)[atomic_numbers - 1]
    return k_hot_mask


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask logits by setting the logits for masked items to -inf.

    Args:
        logits (torch.Tensor): Logits of shape (batch_size, num_classes)
        mask (torch.Tensor): Mask of shape (batch_size, num_classes). Values with zero are masked.

    Returns:
        torch.Tensor: Masked logits
    """
    return logits + (1 - mask) * -1e10


def mask_disallowed_elements(
    logits: torch.FloatTensor,
    x: ChemGraph | None = None,
    batch_idx: torch.LongTensor | None = None,
    predictions_are_zero_based: bool = True,
):
    """
    Mask out atom types that are disallowed in general,
    as well as potentially all elements not in the chemical system we condition on.

    Args:
        logits (torch.Tensor): Logits of shape (batch_size, num_classes)
        x (ChemGraph)
        batch_idx (torch.LongTensor, optional): Batch indices. Defaults to None. Must be provided if condition is not None.
        predictions_are_zero_based (bool, optional): Whether the logits are zero-based. Defaults to True. Basically, if we're using D3PM,
            the logits are zero-based (model predicts atomic number index)
    """
    # First, mask out generally undesired elements
    # (1, num_selected_elements)
    selected_atomic_numbers = torch.tensor(SELECTED_ATOMIC_NUMBERS, device=logits.device)
    predictions_are_one_based = not predictions_are_zero_based
    # (num_atoms, num_classes)
    one_hot_selected_elements = atomic_numbers_to_mask(
        atomic_numbers=selected_atomic_numbers + int(predictions_are_one_based),
        max_atomic_num=logits.shape[1],
    )
    # (1, num_classes)
    k_hot_mask = one_hot_selected_elements.sum(0)[None]
    # Set the logits for disallowed elements to -inf
    logits = mask_logits(logits=logits, mask=k_hot_mask)

    # Optionally, also mask out elements that are not in the chemical system we condition on
    if x is not None and "chemical_system" in x and x["chemical_system"] is not None:
        try:
            # torch.BoolTensor, shape (batch_size, 1)  -- do not mask logits when we use an unconditional embedding
            do_not_mask_atom_logits = get_use_unconditional_embedding(
                batch=x, cond_field="chemical_system"
            )
        except KeyError:
            # if no mask provided to use conditional/unconditional labels then do not mask logits
            do_not_mask_atom_logits = torch.ones(
                (len(x["chemical_system"]), 1), dtype=torch.bool, device=x["num_atoms"].device
            )

        # mypy
        assert batch_idx is not None, "batch_idx must be provided if condition is not None"
        # Only mask atom types where the condition is not masked
        # A 1 means that we do not alter the logit, a 0 means that we change the logit to -inf
        # keep_logits.shape=(Nbatch, MAX_ATOMIC_NUM+1)

        # 1 = keep logit, 0 = set logit to -inf, shape = (Nbatch, MAX_ATOMIC_NUM+1)
        keep_all_logits = torch.ones((len(x["chemical_system"]), 1), device=x["num_atoms"].device)

        # torch.Tensor, shape=(Nbatch,MAX_ATOMIC_NUM+1) -- 1s where elements are present in chemical system condition, 0 elsewhere
        multi_hot_chemical_system = ChemicalSystemMultiHotEmbedding.sequences_to_multi_hot(
            x=ChemicalSystemMultiHotEmbedding.convert_to_list_of_str(x=x["chemical_system"]),
            device=x["num_atoms"].device,
        )

        keep_logits = torch.where(
            do_not_mask_atom_logits,
            keep_all_logits,
            multi_hot_chemical_system,
        )
        # This is converting the 1-based chemical system condition to a 0-based
        # condition -- we're doing it on the multi-hot representation of the
        # chemical system, so we need to shift the indices by one.
        if predictions_are_zero_based:
            keep_logits = keep_logits[:, 1:]
            # If we use mask diffusion, logits is shape [batch_size, MAX_ATOMIC_NUM + 1]
            # instead of [batch_size, MAX_ATOMIC_NUM], so we have to add one dummy column
            if keep_logits.shape[1] == logits.shape[1] - 1:
                keep_logits = torch.cat([keep_logits, torch.zeros_like(keep_logits[:, :1])], dim=-1)
        # Mask out all logits outside the chemical system we condition on
        logits = mask_logits(logits, keep_logits[batch_idx])

    return logits


def get_chemgraph_from_denoiser_output(
    pred_atom_types: torch.Tensor,
    pred_lattice_eps: torch.Tensor,
    pred_cart_pos_eps: torch.Tensor,
    training: bool,
    element_mask_func: Callable | None,
    x_input: ChemGraph,
) -> ChemGraph:
    """
    Convert raw denoiser output to ChemGraph and optionally apply masking to element logits.

    Keyword arguments
    -----------------
    pred_atom_atoms: predicted logits for atom types
    pred_lattice_eps: predicted lattice noise
    pred_cart_pos_eps: predicted cartesian position noise
    training: whether or not the model is in training mode - logit masking is only applied when sampling
    element_mask_func: when not training, a function can be applied to mask logits for certain atom types
    x_input: the nosiy state input to the score model, contains the lattice to convert cartesisan to fractional noise.
    """
    if not training and element_mask_func:
        # when sampling we may want to mask logits for atom types depending on info in x['chemical_system'] and x['chemical_system_MASK']
        pred_atom_types = element_mask_func(
            logits=pred_atom_types,
            x=x_input,
            batch_idx=x_input.get_batch_idx("pos"),
        )

    replace_dict = dict(
        # convert from cartesian to fractional coordinate score
        pos=(
            x_input["cell"].inverse().transpose(1, 2)[x_input.get_batch_idx("pos")]
            @ pred_cart_pos_eps.unsqueeze(-1)
        ).squeeze(-1),
        cell=pred_lattice_eps,
        atomic_numbers=pred_atom_types,
    )
    return x_input.replace(
        **replace_dict,
    )


class GemNetTDenoiser(ScoreModel):
    """Denoiser"""

    def __init__(
        self,
        gemnet: nn.Module,
        hidden_dim: int = 512,
        denoise_atom_types: bool = True,
        atom_type_diffusion: str = [
            "mask",
            "uniform",
        ][0],
        property_embeddings: torch.nn.ModuleDict | None = None,
        property_embeddings_adapt: torch.nn.ModuleDict | None = None,
        element_mask_func: Callable | None = None,
        **kwargs,
    ):
        """Construct a GemNetTDenoiser object.

        Args:
            gemnet: a GNN module
            hidden_dim (int, optional): Number of hidden dimensions in the GemNet. Defaults to 128.
            denoise_atom_types (bool, optional): Whether to denoise the atom  types. Defaults to False.
            atom_type_diffusion (str, optional): Which type of atom type diffusion to use. Defaults to "mask".
            condition_on (Optional[List[str]], optional): Which aspects of the data to condition on. Strings must be in ["property", "chemical_system"]. If None (default), condition on ["chemical_system"].
        """
        super(GemNetTDenoiser, self).__init__()

        self.gemnet = gemnet
        self.noise_level_encoding = NoiseLevelEncoding(hidden_dim)
        self.hidden_dim = hidden_dim
        self.denoise_atom_types = denoise_atom_types
        self.atom_type_diffusion = atom_type_diffusion

        # torch.nn.ModuleDict: Dict[PropertyName, PropertyEmbedding]
        self.property_embeddings = torch.nn.ModuleDict(property_embeddings or {})

        with_mask_type = self.denoise_atom_types and "mask" in self.atom_type_diffusion
        self.fc_atom = nn.Linear(hidden_dim, MAX_ATOMIC_NUM + int(with_mask_type))

        self.element_mask_func = element_mask_func

    def forward(self, x: ChemGraph, t: torch.Tensor) -> ChemGraph:
        """
        args:
            x: tuple containing:
                frac_coords: (N_atoms, 3)
                lattice: (N_cryst, 3, 3)
                atom_types: (N_atoms, ), need to use atomic number e.g. H = 1 or ion state
                num_atoms: (N_cryst,)
                batch: (N_atoms,)
            t: (N_cryst,): timestep per crystal
        returns:
            tuple of:
                predicted epsilon: (N_atoms, 3)
                lattice update: (N_crystals, 3, 3)
                predicted atom types: (N_atoms, MAX_ATOMIC_NUM)
        """
        (frac_coords, lattice, atom_types, num_atoms, batch) = (
            x["pos"],
            x["cell"],
            x["atomic_numbers"],
            x["num_atoms"],
            x.get_batch_idx("pos"),
        )

        # (num_atoms, hidden_dim) (num_crysts, 3)
        t_enc = self.noise_level_encoding(t).to(lattice.device)
        z_per_crystal = t_enc

        # evaluate property embedding values
        property_embedding_values = get_property_embeddings(
            batch=x, property_embeddings=self.property_embeddings
        )

        if len(property_embedding_values) > 0:
            z_per_crystal = torch.cat([z_per_crystal, property_embedding_values], dim=-1)

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
        return list(self.property_embeddings)
