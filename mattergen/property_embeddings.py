# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Sequence, Union

import torch

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.types import PropertySourceId, TargetProperty
from mattergen.common.utils.data_utils import get_atomic_number
from mattergen.common.utils.globals import MAX_ATOMIC_NUM, PROPERTY_SOURCE_IDS

# attribute name in ChemGraph corresponding to a Dict[PropertyName, torch.BoolTensor]
# object that stores whether to use the unconditional embedding for each conditional field
_USE_UNCONDITIONAL_EMBEDDING = "_USE_UNCONDITIONAL_EMBEDDING"


def replace_use_unconditional_embedding(
    batch: ChemGraph, use_unconditional_embedding: Dict[PropertySourceId, torch.BoolTensor]
) -> ChemGraph:
    """
    Set the use of conditional or unconditional embeddings for each conditional field in the batch.
    This utility will overwrite any batch._USE_CONDITIONAL_EMBEDDING keys included in use_unconditional_embedding
    but will keep the value of any keys in batch._USE_CONDITIONAL_EMBEDDING that are not in
    use_unconditional_embedding.

    Keyword arguments
    -----------------
    batch: ChemGraph -- the batch of data to be modified.
    use_unconditional_embedding: Dict[PropertyName, torch.BoolTensor] -- a dictionary whose values
        are torch.BoolTensors of shape (n_structures_in_batch, 1) stating whether to use the unconditional embedding for
        each conditional field. The keys are the names of the conditional fields in the batch.


    Returns
    -------
    ChemGraph -- the modified batch of data containing
        ChemGraph._USE_CONDITIONAL_EMBEDDING: Dict[PropertyName, torch.BoolTensor]. When
        ChemGraph[_USE_UNCONDITIONAL_EMBEDDING][cond_field][ii] is True, the iith data point will
        use its unconditional embedding for cond_field. When False, the conditional embedding will be used.
    """
    try:
        existing_use_unconditional_embedding = batch[_USE_UNCONDITIONAL_EMBEDDING]

        for k, v in use_unconditional_embedding.items():
            existing_use_unconditional_embedding[k] = v

        return batch.replace(**{_USE_UNCONDITIONAL_EMBEDDING: existing_use_unconditional_embedding})
    except KeyError:
        # no existing data
        return batch.replace(**{_USE_UNCONDITIONAL_EMBEDDING: use_unconditional_embedding})


def get_use_unconditional_embedding(
    batch: ChemGraph, cond_field: PropertySourceId
) -> torch.BoolTensor:
    """
    Returns
    -------
    torch.BoolTensor, shape=(n_structures_in_batch, 1) -- whether to use the unconditional embedding for cond_field.
        When True, we use unconditional embedding.

    NOTE: When _USE_UNCONDITIONAL_EMBEDDING is not in ChemGraph or cond_field is not
        in ChemGraph[_USE_UNCONDITIONAL_EMBEDDING] we return a torch.BoolTensor with False
        values. This allows a model trained conditional data to evaluate an unconditional score
        without having to specify any conditional data in ChemGraph.
    """
    try:
        return batch[_USE_UNCONDITIONAL_EMBEDDING][cond_field]
    except KeyError:
        # when a PropertyEmbedding exists for a conditional field but it is
        # not present in the ChemGraph, SetUnconditionalEmbeddingType and
        # SetConditionalEmbeddingType will fail to set the torch.BoolTensor that
        # get_use_conditional_embedding looks for. This results in a KeyError
        # which we interpret as the user wanting to use the unconditional
        # embedding for this property.
        return torch.ones_like(batch["num_atoms"], dtype=torch.bool).reshape(-1, 1)


def tensor_is_not_nan(x: torch.Tensor) -> torch.BoolTensor:
    """
    Keyword arguments
    -----------------
    x: torch.Tensor, shape = (n_structures_in_batch, Ndim) -- labels for a single conditional field.
        We assume that when a label is not present, the corresponding value is specified
        as torch.nan.

    Returns
    -------
    torch.BoolTensor, shape = (n_structures_in_batch,) -- index i is True if x[i] contains no NaNs
    """
    return torch.all(
        torch.reshape(torch.logical_not(torch.isnan(x)), (x.shape[0], -1)),
        dim=1,
    )


def data_is_not_nan(
    x: Union[torch.Tensor, list[str | None], list[list[str] | None]]
) -> torch.BoolTensor:
    """
    Returns (n_structures_in_batch,) torch.BoolTensor of whether the conditional values
    for a given property are not nan.

    NOTE: Currently we enforce no restriction on the data type that properties can have in
    ChemGraph. The intent is that ChemGraph always contains property values in their
    representation and type seen by the user. This means however that we have to distribute
    handling of different types throughout the code, this function is one such place.

    """
    if isinstance(x, torch.Tensor):
        return tensor_is_not_nan(x=x)
    else:
        return torch.tensor([_x is not None for _x in x])


def get_cond_field_names_in_batch(x: ChemGraph) -> list[str]:
    """
    Returns a list of field names that are known to be conditional properties in
    PROPERTY_SOURCE_IDS, which are present in x.
    """
    return [str(k) for k in x.keys() if k in PROPERTY_SOURCE_IDS]


class SetEmbeddingType:
    def __init__(
        self,
        p_unconditional: float,
        dropout_fields_iid: bool = False,
    ):
        """
        In PropertyEmbedding.forward we choose to concatenate either an unconditional embedding
        (ignores the value of a property) or a conditional embedding (depends on the value of a property)
        to the tensor that is input to the first node layer of each atom. This utility sets the internal state
        of ChemGraph to randomly select either the conditional or unconditional embedding for each structure
        in the batch.

        ChemGraph.[_USE_UNCONDITIONAL_EMBEDDING]: boolTensor, shape=(n_structures_in_batch, 1) stores a True
        value for structures where we intend to use the unconditional embedding for all atoms contained in
        that corresponding structure.

        This utility operates in 2 modes:
        1) dropout_fields_iid = True -- We randomly assign which conditional fields are unconditional and which
            are conditional for fields that are not nan independently of whether all conditional fields are not
            nan for that structure. This means that for a structure conditioned on (y1,y2) we can generate embeddings
            corresponding to p(x), p(x|y1), p(x|y2), p(x|y1,y2).
        2) dropout_fields_iid = False - We assign conditional or unconditional embeddings to all conditional fields
            of a single structure simultaneously. This means that for a structure conditioned on (y1,y2) we can
            only generate embeddings corresponding to p(x) and p(|y1,y2).

        Keyword args:
        -------------
        p_unconditional: float -- the probability of using the unconditional embedding in the score model.
        dropout_fields_iid: bool -- whether to mask the conditional embedding of fields independently and
            identically distributed according to p_unconditional. If False, the score model is only exposed
            to two scenarios: 1) all conditional fields have their unconditional embedding. 2) all conditional
            fields have their conditional embedding. If True, the score model is exposed to all possible
            combinations of conditional fields having their unconditional or conditional embeddings, ie the score
            model will learn p(x), p(x|y1), p(x_y2), p(x|y1,y2),...

            Note: when dropout_fields_iid=False, the conditional embedding will only be used when all
            conditional fields have data present. If no single data point has data present for all conditional
            fields, then the score model will only be exposed to the unconditional embedding state p(x) and the
            joint p(x|y1,y2,...) will not be learned.
        """
        self.p_unconditional = p_unconditional
        self.dropout_fields_iid = dropout_fields_iid

    def __call__(self, x: ChemGraph) -> ChemGraph:
        # list of conditional fields present in the batch
        cond_fields: list[str] = get_cond_field_names_in_batch(x=x)

        if len(cond_fields) == 0:
            return x
        else:
            # assume all conditional fields have same batch size
            batch_size = len(x[cond_fields[0]])

            # not all cond_fields are torch tensor objects, eg chemical_system
            device = x["num_atoms"].device

            # get dictionary of which conditional fields are present (not nan)
            # values are torch.BoolTensors of shape (batch_size, ) - when element 'i' is True, a label exists for this data point for this field
            data_is_not_nan_dict: Dict[PropertySourceId, torch.BoolTensor] = {
                cond_field: data_is_not_nan(x=x[cond_field]).to(device=device)  # type: ignore
                for cond_field in cond_fields
            }

            # element `i` is True when all conditional fields have data present for this data point
            # this is useful for when we want to use the (un)conditional embedding for all conditional
            # fields per data point simultaneously
            alldata_is_not_nan: torch.BoolTensor = torch.all(
                torch.cat(
                    [
                        cond_data_not_nan.reshape(-1, 1)
                        for cond_data_not_nan in data_is_not_nan_dict.values()
                    ],
                    dim=1,
                ),
                dim=1,
            )

            # when True, use the unconditional embedding for this conditional field and data point
            use_unconditional_embedding: Dict[PropertySourceId, torch.BoolTensor] = {}

            for cond_field in cond_fields:
                # by default use the unconditional embedding (embedding_type=True)
                embedding_type = torch.ones((batch_size, 1), device=device, dtype=torch.bool)

                if self.dropout_fields_iid:
                    # torch.BoolTensor, shape = (n_structures_in_batch, 1) -- True when conditional field is not nan
                    cond_data_is_not_nan = data_is_not_nan_dict[cond_field]  # type: ignore
                else:
                    # torch.BoolTensor, shape = (n_structures_in_batch, 1) -- True when all conditional fields are not nan
                    cond_data_is_not_nan = alldata_is_not_nan

                # assign conditional embedding to (1-self.p_unconditional) of values where cond_data_is_not_nan=True
                embedding_type[cond_data_is_not_nan] = (  # type: ignore
                    torch.rand((cond_data_is_not_nan.sum(), 1), device=device)  # type: ignore
                    <= self.p_unconditional
                )

                # torch.BoolTensor, shape=(n_structures_in_batch,1) -- when True use the unconditional embedding
                use_unconditional_embedding[cond_field] = embedding_type  # type: ignore

            return replace_use_unconditional_embedding(
                batch=x, use_unconditional_embedding=use_unconditional_embedding
            )


class SetUnconditionalEmbeddingType:
    """
    In PropertyEmbedding.forward we choose to concatenate either an unconditional embedding
    (ignores the value of a property) or a conditional embedding (depends on the value of a property)
    to the tensor that is input to the first node layer of each atom. This utility sets the internal state
    of ChemGraph to use the unconditional embedding for all structures for all conditional fields present
    in the batch. Note that conditional fields in the batch are automatically determined by the presence
    of any PropertyName in ChemGraph.

    ChemGraph.[_USE_UNCONDITIONAL_EMBEDDING]: boolTensor, shape=(n_structures_in_batch, 1) stores True
    for all structures for all conditional properties present in ChemGraph.

    NOTE: If a conditional property was trained on by the model but is not
    specified in the batch, then it will be attributed an unconditional embedding
    in mattergen.property_embeddings.PropertyEmbedding.forward.
    This behaviour allows unconditional samples to be drawn from a model that was trained
    on certain conditions, without having to set any conditional values in ChemGraph.
    """

    def __call__(self, x: ChemGraph) -> ChemGraph:
        # list of conditional fields present in the batch
        cond_fields = get_cond_field_names_in_batch(x=x)

        device = x["num_atoms"].device

        return replace_use_unconditional_embedding(
            batch=x,
            use_unconditional_embedding={
                cond_field: torch.ones((len(x[cond_field]), 1), dtype=torch.bool, device=device)  # type: ignore
                for cond_field in cond_fields
            },
        )


class SetConditionalEmbeddingType:
    """
    In PropertyEmbedding.forward we choose to concatenate either an unconditional embedding
    (ignores the value of a property) or a conditional embedding (depends on the value of a property)
    to the tensor that is input to the first node layer of each atom. This utility sets the internal state
    of ChemGraph to use the unconditional embedding for all structures for all conditional fields present
    in the batch. Note that conditional fields in the batch are automatically determined by the presence
    of any PropertyName on in ChemGraph.

    ChemGraph.[_USE_UNCONDITIONAL_EMBEDDING]: boolTensor, shape=(n_structures_in_batch, 1) stores False
    for all structures for all conditional properties present in ChemGraph.

    NOTE: If a conditional property was trained on by the model but is not
    specified in the batch, then it will be attributed an unconditional embedding
    in mattergen.property_embeddings.PropertyEmbedding.forward.
    This behaviour allows unconditional samples to be drawn from a model that was trained
    on certain conditions, without having to set any conditional values in ChemGraph.
    """

    def __call__(self, x: ChemGraph) -> ChemGraph:
        # a list of all conditional properties present in the batch
        cond_fields = get_cond_field_names_in_batch(x=x)

        device = x["num_atoms"].device

        use_unconditional_embedding = {}
        for cond_field in cond_fields:
            # use the conditional embedding for all conditional
            # properties present in the batch. If we want to sample
            # marginalise out any conditional properties that the model
            # was trained on, them exclude them from ChemGraph
            use_unconditional_embedding[cond_field] = torch.zeros(
                (len(x[cond_field]), 1), dtype=torch.bool, device=device
            )

        return replace_use_unconditional_embedding(
            batch=x, use_unconditional_embedding=use_unconditional_embedding  # type: ignore
        )


class BaseUnconditionalEmbeddingModule(torch.nn.Module):
    # If True, we don't need conditional values to evaluate an unconditional score
    # This allows evaluationg of an unconditional score without needing to specify
    # any conditional values in the batch
    only_depends_on_shape_of_input: bool

    # This is the embedding dimension, the embedding module will output a
    # torch.tensor of shape (n_structures_in_batch, hidden_dim)
    hidden_dim: int


class EmbeddingVector(BaseUnconditionalEmbeddingModule):
    # If True, we don't need conditional values to evaluate an unconditional score
    only_depends_on_shape_of_input: bool = True

    def __init__(self, hidden_dim: int):
        super().__init__()
        # a vector of learnable parameters of shape (hidden_dim,)
        self.embedding = torch.nn.Embedding(1, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This forward depends only on the shape of x and returns a tensor of zeros.
        """
        return self.embedding(
            torch.zeros(len(x), dtype=torch.long, device=self.embedding.weight.device)
        )


class SpaceGroupEmbeddingVector(BaseUnconditionalEmbeddingModule):
    # If True, we don't need conditional values to evaluate an unconditional score
    only_depends_on_shape_of_input: bool = True

    def __init__(self, hidden_dim: int):
        super().__init__()
        # a vector of learnable parameters of shape (hidden_dim,)
        self.embedding = torch.nn.Embedding(230, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return embedding of the space group, 1 is subtracted from the space group number to
        make it zero-indexed.
        """
        return self.embedding(x.long() - 1)


class ZerosEmbedding(BaseUnconditionalEmbeddingModule):
    """
    Return a [n_crystals_in_batch, self.hidden_dim] tensor of zeros. This is helpfuln as the unconditional embedding
    for a property included in the adapter module if we do not want to change the unconditional score
    of the base model when properties are added in the adapter module.
    """

    # If True, we don't need conditional values to evaluate an unconditional score
    only_depends_on_shape_of_input: bool = True

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor | list[str]) -> torch.Tensor:
        """
        This forward depends only on the shape of x.
        """
        return torch.zeros(len(x), self.hidden_dim)


class ChemicalSystemMultiHotEmbedding(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Linear(in_features=MAX_ATOMIC_NUM + 1, out_features=hidden_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _sequence_to_multi_hot(x: Sequence[str], device: torch.device) -> torch.Tensor:
        """
        Converts a sequence of unique elements present in a single structure to a multi-hot
        vectors of 1s (present) and 0s (not present) for each unique element.

        Returns
        -------
        torch.Tensor, shape = (1, MAX_ATOMIC_NUM + 1)
        """
        # torch.LongTensor of indices of each element in the sequence
        chemical_system_numbers: torch.LongTensor = torch.tensor(
            [get_atomic_number(symbol=_element) for _element in x], dtype=int, device=device
        )
        # 1-d vectors of 1s and 0s for each unique element
        chemical_system_condition = torch.zeros(MAX_ATOMIC_NUM + 1, device=device)
        # set 1s for elements that are present
        chemical_system_condition[chemical_system_numbers] = 1.0
        return chemical_system_condition.reshape(1, -1)

    @staticmethod
    def sequences_to_multi_hot(x: list[list[str]], device: torch.device) -> torch.Tensor:
        """
        Convert a list of sequences of unique elements present in a list of structures to a multi-hot
        tensor of 1s (present) and 0s (not present) for each unique element.

        Returns
        -------
        torch.Tensor, shape = (n_structures_in_batch, MAX_ATOMIC_NUM + 1)
        """
        return torch.cat(
            [ChemicalSystemMultiHotEmbedding._sequence_to_multi_hot(_x, device=device) for _x in x],
            dim=0,
        )

    @staticmethod
    def convert_to_list_of_str(x: list[str] | list[list[str]]) -> list[list[str]]:
        """
        Returns
        -------
        list[list[str]] -- a list of length n_structures_in_batch of chemical systems for each structure
            where the chemical system is specified as a list of unique elements in the structure.
        """
        if isinstance(x[0], str):
            # list[Sequence[str]]
            x = [_x.split("-") for _x in x if isinstance(_x, str)]

        return x  # type: ignore

    def forward(self, x: list[str] | list[list[str]]) -> torch.Tensor:
        """
        Keyword arguments
        -----------------
        x: Union[list[str], list[Sequence[str]]] -- if elements are a string, they are assumed to be
            a '-' delimited list of unique elements. If a sequence of strings, it is assumed to be a list of
            unique elements in the structure.
        """
        # make sure each chemical system is specified as a list of unique elements in the structure
        # list[list[str]]
        x = self.convert_to_list_of_str(x=x)

        # shape=(n_structures_in_batch, MAX_ATOMIC_NUM + 1)
        multi_hot_representation: torch.Tensor = self.sequences_to_multi_hot(x=x, device=self.device)  # type: ignore

        return self.embedding(multi_hot_representation)


class PropertyEmbedding(torch.nn.Module):
    def __init__(
        self,
        name: PropertySourceId,
        conditional_embedding_module: torch.nn.Module,
        unconditional_embedding_module: BaseUnconditionalEmbeddingModule,
        scaler: torch.nn.Module = torch.nn.Identity(),
    ):
        super().__init__()
        self.name = name
        self.conditional_embedding_module = conditional_embedding_module
        self.unconditional_embedding_module = unconditional_embedding_module
        self.scaler = scaler
        assert self.name in PROPERTY_SOURCE_IDS, (
            f"PropertyEmbedding.name {self.name} not found in the database. "
            f"Available property_source_ids: {PROPERTY_SOURCE_IDS}"
        )

    def forward(self, batch: ChemGraph) -> torch.Tensor:
        """
        ChemGraph[_USE_UNCONDITIONAL_EMBEDDING]: Dict[str, torch.BoolTensor]
        has values torch.BoolTensor, shape=(n_structures_in_batch, 1) that when True, denote that
        we should use the unconditional embedding (instead of the conditional embedding) as input
        for that property to the input nodes of each atom in the structure.

        In this forward, we return a torch.Tensor, shape=(n_structures_in_batch, hidden_dim) of
        embedding values for this property for each structure in the batch. Based on the state of
        ChemGraph[_USE_UNCONDITIONAL_EMBEDDING] we return either the unconditional or conditional
        embedding for each element i in torch.Tensor[i].

        NOTE: when self.name is not in ChemGraph[_USE_UNCONDITIONAL_EMBEDDING] we apply the
        unconditional embedding. This is to adopt the behaviour that when no conditional value is
        specified in ChemGraph, a model that was trained on said property will generate an
        unconditional score.
        """
        # shape=(n_structures_in_batch, 1) -- True when use the unconditional embedding
        # NOTE: when ChemGraph[_USE_UNCONDITIONAL_EMBEDDING][self.name] is absent, as
        # happens when self.name is missing from ChemGraph, we return a torch.BoolTensor
        # that is all True - ie. we use the unconditional embedding for all structures
        # in the batch. This is so that we can draw unconditional samples from a model
        # trained on conditions without having to specify conditional values in ChemGraph.
        use_unconditional_embedding: torch.BoolTensor = get_use_unconditional_embedding(
            batch=batch, cond_field=self.name
        )

        if (
            torch.all(use_unconditional_embedding)
            and self.unconditional_embedding_module.only_depends_on_shape_of_input
        ):
            # this allows evaluation of the unconditional score without having to supply conditional values for this property
            return self.unconditional_embedding_module(x=batch["num_atoms"]).to(batch.pos.device)
        else:
            # raw values for the conditional data as seen by the user, eg dft_bulk_modulus=torch.tensor([300]*n_structures_in_batch)
            data = batch[self.name]
            if isinstance(data, torch.Tensor) and data.dim() == 2:
                # [B, 1] => [B,]
                data = data.squeeze(-1)

            # optionally apply normalization, eg unit standard deviation and zero mean
            data = self.scaler(data)
            conditional_embedding: torch.Tensor = self.conditional_embedding_module(data)
            unconditional_embedding: torch.Tensor = self.unconditional_embedding_module(x=data).to(
                batch.pos.device
            )

            return torch.where(
                use_unconditional_embedding, unconditional_embedding, conditional_embedding
            )

    def fit_scaler(self, all_data):
        if isinstance(self.scaler, torch.nn.Identity):
            return
        self.scaler.fit(all_data)


def get_property_embeddings(
    batch: ChemGraph, property_embeddings: torch.nn.ModuleDict
) -> torch.Tensor:
    """
    Keyword arguments
    -----------------
    property_embeddings: torch.nn.ModuleDict[PropertyToConditonOn, PropertyEmbedding] -- a dictionary
        of property embeddings. The keys are the names of the conditional fields in the batch.
    """
    # we need a consistent order for the embeddings that does not depend on the order
    # specified by the user
    ordered_keys = sorted(property_embeddings.keys())

    if len(ordered_keys) > 0:
        # shape = (n_structures_in_batch, sum(embedding_dims)) for embedding_dims: list[int] a list of the output dimension for each embedding
        return torch.cat(
            [property_embeddings[k].forward(batch=batch) for k in ordered_keys], dim=-1
        )
    else:
        # torch.cat doesn't accept an empty list
        return torch.tensor([], device=batch["num_atoms"].device)


def set_conditional_property_values(batch: ChemGraph, properties: TargetProperty) -> ChemGraph:
    # list of conditional field names that are not torch tensor objects in ChemGraph
    not_numeric = [k for k, v in properties.items() if not isinstance(v, (int, float))]

    cond_values = {
        k: (
            [properties[k]] * len(batch["num_atoms"])
            if k in not_numeric
            else torch.full_like(batch["num_atoms"], v).reshape(-1, 1)
        )
        for k, v in properties.items()
    }

    return batch.replace(**cond_values)  # type: ignore
