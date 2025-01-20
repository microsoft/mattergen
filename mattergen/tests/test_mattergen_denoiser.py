# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import torch
from torch_scatter import scatter_add

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.transform import set_chemical_system_string
from mattergen.common.utils.globals import MAX_ATOMIC_NUM
from mattergen.denoiser import mask_disallowed_elements
from mattergen.property_embeddings import (
    ChemicalSystemMultiHotEmbedding,
    SetConditionalEmbeddingType,
    SetEmbeddingType,
    SetUnconditionalEmbeddingType,
    get_use_unconditional_embedding,
    replace_use_unconditional_embedding,
)


@pytest.mark.parametrize("p_unconditional", [0.0, 1.0])
def test_pre_corruption_fn(p_unconditional: float):
    pre_corruption_fn = SetEmbeddingType(
        p_unconditional=p_unconditional, dropout_fields_iid=False  # type: ignore
    )

    # dummy data with nan (missing data) values in batch elements 0, 2, 3
    pos = torch.rand((10, 2, 3))

    # if any element of a data point is nan, whole data point
    # will be treated as having missing label
    pos[0, 0, 0] = torch.nan
    pos[2, 1, 2] = torch.nan
    pos[3] = torch.nan

    x_with_mask = pre_corruption_fn(
        x=ChemGraph(pos=pos, num_atoms=torch.ones((10, 1), dtype=int), dft_bulk_modulus=pos)
    )

    mask = get_use_unconditional_embedding(batch=x_with_mask, cond_field="dft_bulk_modulus")  # type: ignore

    # there are 7 available labels in total, corresponding to 3 masked conditions
    num_masked = {0.0: 3, 1.0: 10}[p_unconditional]

    assert mask.sum() == num_masked


@pytest.mark.parametrize(
    "p_unconditional, dropout_fields_iid", [(0.0, True), (0.0, False), (1.0, True), (1.0, False)]
)
def test_pre_corruption_fn_multi(p_unconditional: float, dropout_fields_iid: bool):
    pre_corruption_fn = SetEmbeddingType(
        p_unconditional=p_unconditional,
        dropout_fields_iid=dropout_fields_iid,
    )

    # dummy data with nan (missing data) values in batch elements 0, 2, 3
    pos = torch.rand((10, 2, 3))

    # if any element of a data point is nan, whole data point
    # will be treated as having missing label
    pos[0, 0, 0] = torch.nan
    pos[2, 1, 2] = torch.nan
    pos[3] = torch.nan

    # dummy data with nan (missing data) values in batch elements 1, 2, 4, 5
    cell = torch.rand((10, 3, 3))

    cell[1, 0, 0] = torch.nan
    cell[2, 1, 0] = torch.nan
    cell[4, 0, 2] = torch.nan
    cell[5, 1, 1] = torch.nan

    x_with_mask = pre_corruption_fn(
        x=ChemGraph(
            pos=pos,
            num_atoms=torch.ones((10, 1), dtype=int),
            cell=cell,
            dft_bulk_modulus=pos,
            dft_shear_modulus=cell,
        )
    )

    for cond_field in ["dft_bulk_modulus", "dft_shear_modulus"]:
        # True when we want to use unconditional embedding
        mask = get_use_unconditional_embedding(batch=x_with_mask, cond_field=cond_field)  # type: ignore

        number_masked = mask.sum()

        if p_unconditional == 0.0:
            # all conditional embeddings are used for non-nan data

            if dropout_fields_iid:
                # conditional fields are masked independently
                expected_number_masked = {"dft_bulk_modulus": 3, "dft_shear_modulus": 4}[cond_field]
            else:
                # all conditional fields must be non nan
                expected_number_masked = 6
        elif p_unconditional == 1.0:
            # no conditional embeddings are used
            expected_number_masked = 10
        else:
            raise Exception("p_unconditional must be 0.0 or 1.0")

        assert number_masked == expected_number_masked

    # test SetEmbeddingType runs successfully when no conditioning fields are specified
    pre_corruption_fn = SetEmbeddingType(
        p_unconditional=p_unconditional,
        dropout_fields_iid=dropout_fields_iid,
    )

    _ = pre_corruption_fn(
        x=ChemGraph(
            pos=pos,
            cell=cell,
            dft_bulk_modulus=pos,
            dft_shear_modulus=cell,
            num_atoms=torch.ones((10, 1)),
        )
    )


def test_remove_conditioning_fn():
    # check all relevant fields are masked

    x = ChemGraph(
        pos=torch.rand(10, 3),
        forces=torch.rand(10, 3),
        atomic_numbers=torch.ones((10,), dtype=torch.int),
        num_atoms=torch.ones((10, 1), dtype=int),
        dft_bulk_modulus=torch.randn(10, 3),
        dft_shear_modulus=torch.randn(10, 3),
    )

    # fields to condition on
    cond_fields = ["dft_bulk_modulus", "dft_shear_modulus"]

    # introduce masking attribute that is always True for conditioned on fields
    mask_all = SetUnconditionalEmbeddingType()

    x_with_mask = mask_all(x=x)

    for cond_field in cond_fields:
        torch.testing.assert_close(
            get_use_unconditional_embedding(batch=x_with_mask, cond_field=cond_field),  # type: ignore
            torch.ones((10, 1), dtype=torch.bool),
        )


def test_keep_conditioning_fn():
    x = ChemGraph(
        pos=torch.rand(10, 3),
        forces=torch.rand(10, 3),
        atomic_numbers=torch.ones((10,), dtype=torch.int),
        num_atoms=torch.ones((10, 1), dtype=int),
        dft_bulk_modulus=torch.rand(10, 3),
        dft_shear_modulus=torch.randn(10, 3),
    )

    # fields to condition on

    # mask atomic_numbers as we don't want to condition on this
    x_with_mask = SetConditionalEmbeddingType()(x=x)

    # we do not want to condition on atomic_numbers, so they are masked
    torch.testing.assert_close(
        get_use_unconditional_embedding(batch=x_with_mask, cond_field="dft_bulk_modulus"),
        torch.zeros((10, 1), dtype=torch.bool),
    )

    # we do want to condition on pos, so it is not masked
    torch.testing.assert_close(
        get_use_unconditional_embedding(batch=x_with_mask, cond_field="dft_shear_modulus"),
        torch.zeros((10, 1), dtype=torch.bool),
    )


@pytest.mark.parametrize("zero_based_predictions", [True, False])
def test_mask_disallowed_elements(zero_based_predictions: bool):
    torch.manual_seed(23232)
    samples = [
        ChemGraph(
            pos=torch.rand(10, 3),
            num_atoms=torch.tensor([10]),
            atomic_numbers=6 * torch.ones((10,), dtype=torch.int),
            cell=torch.eye(3),
        ),
        ChemGraph(
            pos=torch.rand(5, 3),
            num_atoms=torch.tensor([5]),
            # La, Na, O, Sb, Sc
            atomic_numbers=torch.tensor([57, 11, 8, 51, 21]),
            cell=torch.eye(3),
        ),
        ChemGraph(
            pos=torch.rand(15, 3),
            num_atoms=torch.tensor([15]),
            # La, Na, O, Sb, Sc
            atomic_numbers=torch.tensor([57, 11, 8, 51, 21, 57, 11, 8, 51, 21, 57, 11, 8, 51, 21]),
            cell=torch.eye(3),
        ),
    ]

    transform = set_chemical_system_string
    batch = collate([transform(sample) for sample in samples])
    assert hasattr(batch, "chemical_system")  # mypy
    assert hasattr(batch, "pos")  # mypy
    assert hasattr(batch, "batch")  # mypy
    assert hasattr(batch, "cell")  # mypy
    assert hasattr(batch, "atomic_numbers")  # mypy
    assert hasattr(batch, "num_atoms")  # mypy
    mask = torch.tensor([0, 0, 1], dtype=torch.bool)[:, None]

    batch_chemgraph = ChemGraph(
        pos=batch.pos,
        cell=batch.cell,
        atomic_numbers=batch.atomic_numbers,
        num_atoms=batch.num_atoms,
        chemical_system=batch.chemical_system,
    )
    batch_chemgraph = replace_use_unconditional_embedding(batch=batch_chemgraph, use_unconditional_embedding={"chemical_system": mask})  # type: ignore

    example_logits = torch.randn(batch.pos.shape[0], MAX_ATOMIC_NUM + 1)
    masked_logits = mask_disallowed_elements(
        logits=example_logits,
        x=batch_chemgraph,
        batch_idx=batch.batch,
        predictions_are_zero_based=zero_based_predictions,
    )
    sampled = torch.distributions.Categorical(logits=masked_logits).sample() + int(
        zero_based_predictions
    )
    sampled_onehot = torch.eye(MAX_ATOMIC_NUM + 1)[sampled]
    sampled_chemical_systems = scatter_add(sampled_onehot, batch.batch, dim=0)

    # shape=(Nbatch, MAX_NUM_ATOMS+1)
    chemsys_multi_hot: torch.LongTensor = ChemicalSystemMultiHotEmbedding.sequences_to_multi_hot(
        x=ChemicalSystemMultiHotEmbedding.convert_to_list_of_str(x=batch.chemical_system),
        device=mask.device,
    )

    for ix, system in enumerate(sampled_chemical_systems):
        sampled_types = system.nonzero()[:, 0].tolist()

        chemsys = chemsys_multi_hot[ix].nonzero()[:, 0].tolist()

        if mask[ix] == 0:
            assert set(sampled_types).difference(set(chemsys)) == set()
        else:
            assert set(sampled_types).difference(set(chemsys)) != set()
