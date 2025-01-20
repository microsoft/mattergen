# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy

import torch
import torch_geometric.data as pyg_data
from torch import IntTensor, LongTensor, Tensor
from torch_geometric import utils
from torch_geometric.typing import OptTensor


class ChemGraph(pyg_data.Data):
    r"""A ChemGraph is a Pytorch Geometric Data object describing a MLPotential molecular graph with atoms in 3D space.
    The data object can hold node-level, and graph-level attributes, as well as (pre-computed) edge information.
    In general, :class:`~torch_geometric.data.Data` tries to mimic the
    behaviour of a regular Python dictionary.
    In addition, it provides useful functionality for analyzing graph
    structures, and provides basic PyTorch tensor functionalities.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    introduction.html#data-handling-of-graphs>`__ for the accompanying
    tutorial.

    Args:
        atomic_numbers (LongTensor, optional): Atomic numbers following ase.Atom, (Unknown=0, H=1) with shape
            :obj:`[num_nodes]`. (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix, only set one position value.
            :obj:`[num_nodes, 3]`. (default: :obj:`None`)
        cell (Tensor, optional): Cell matrix if pbc = True, has shape
            :obj:`[1, 3, 3]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Edge indexes (sender, receiver)
            :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge attributes
            :obj:`[num_edges, num_edge_attr]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes to be stored in the data object.
    """

    def __init__(
        self,
        atomic_numbers: IntTensor | None = None,
        pos: OptTensor = None,
        cell: OptTensor = None,
        edge_index: LongTensor | None = None,
        edge_attr: OptTensor = None,
        **kwargs,
    ):
        super().__init__(x=None, edge_index=edge_index, edge_attr=edge_attr, pos=pos, **kwargs)

        if atomic_numbers is not None:
            self.atomic_numbers = atomic_numbers
        if cell is not None:
            self.cell = cell
        self.__dict__["_frozen"] = True

    def __setattr__(self, attr, value):
        if self.__dict__.get("_frozen", False) and attr not in (
            "_num_graphs",
            "_slice_dict",
            "_inc_dict",
            "_collate_structure",
        ):
            raise AttributeError(
                f"Replacing ChemGraph.{attr} in-place. Consider using the self.replace method to create a shallow copy."
            )
        return super().__setattr__(attr, value)

    def replace(self, **kwargs: OptTensor | str | int | float | list) -> "ChemGraph":
        """Returns a shallow copy of the ChemGraph with updated fields."""
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__["_store"] = copy.copy(self._store)
        for key, value in kwargs.items():
            out._store[key] = value
        out._store._parent = out
        return out

    def get_batch_idx(self, field_name: str) -> LongTensor | None:
        """Used by diffusion library to retrieve batch indices for a given field."""
        assert isinstance(
            self, pyg_data.Batch
        )  # ChemGraphBatch subclass is dynamically defined by PyG
        if field_name == "cell":
            # Graph-level attributes become 'dense' fields where the first dimension is batch dimension.
            return None
        elif field_name in [
            "pos",
            "atomic_numbers",
        ]:
            # per-node attributes
            return self.batch
        else:
            try:
                # This happens if 'follow_batch' kwarg was used when constructing the batch
                return self[f"{field_name}_batch"]
            except KeyError:
                raise NotImplementedError(f"Unable to determine batch index for {field_name}")

    def get_batch_size(self):
        # For diffusion library. Only works if self is a ChemGraphBatch
        assert isinstance(self, pyg_data.Batch)
        return self.num_graphs

    def subgraph(self, subset: Tensor) -> "ChemGraph":
        """
        Returns the induced subgraph given by the node indices :obj:`subset`. If no edge indices are
        present, subsets will only be created for node features.

        Args:
            subset (LongTensor or BoolTensor): The nodes to keep.
        """
        # Check for boolean mask or index array.
        if subset.dtype == torch.bool:
            num_nodes = int(subset.sum())
        else:
            num_nodes = subset.size(0)
            subset = torch.unique(subset, sorted=True)

        # If edge indices are provided, determine subgraph components. Otherwise use only `subset`
        # to select relevant nodes of node attributes.
        if self.edge_index is not None:
            out = utils.subgraph(
                subset,
                self.edge_index,
                relabel_nodes=True,
                num_nodes=self.num_nodes,
                return_edge_mask=True,
            )
            edge_index, _, edge_mask = out
        else:
            edge_index = None
            edge_mask = None

        # Create dictionary of the subsets of all quantities.
        masked_data = {}
        for key, value in self:
            if value is None:
                continue
            if key == "edge_index":
                masked_data[key] = edge_index
            if key == "num_nodes":
                masked_data[key] = num_nodes
            elif self.is_node_attr(key):
                cat_dim = self.__cat_dim__(key, value)
                masked_data[key] = utils.select(value, subset, dim=cat_dim)
            elif self.is_edge_attr(key) and edge_index is not None:
                cat_dim = self.__cat_dim__(key, value)
                masked_data[key] = utils.select(value, edge_mask, dim=cat_dim)

        # Generate final graph.
        data = self.replace(**masked_data)

        return data


# Retrieve a pointer for the DynamicInheritance-based PYG Batch class.
# For typing reasons only, use isinstance(pyg_data.Batch) for runtime checks.
ChemGraphBatch = pyg_data.Batch(_base_cls=ChemGraph).__class__
