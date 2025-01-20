# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
from typing import TypeVar

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from tqdm.auto import tqdm

from mattergen.denoiser import GemNetTDenoiser
from mattergen.diffusion.lightning_module import DiffusionLightningModule
TensorOrStringType = TypeVar("TensorOrStringType", torch.Tensor, list[str])


def maybe_to_tensor(values: list[TensorOrStringType]) -> TensorOrStringType:
    if isinstance(values[0], torch.Tensor):
        return torch.cat(values)
    # chemical system is str and therefore cannot be converted to tensor
    return [el for x in values for el in x]


class SetPropertyScalers(Callback):
    """
    Utility callback; at the start of training, this computes the mean and std of the property data and adds the property
    scalers to the model.
    """

    @staticmethod
    def _compute_property_scalers(
        datamodule: pl.LightningDataModule, property_embeddings: torch.nn.ModuleDict
    ):
        property_values = defaultdict(list)

        # property names may be distinct from keys in this dictionary
        property_names = [p.name for p in property_embeddings.values() if not isinstance(p.scaler, torch.nn.Identity)]
        if len(property_names) == 0:
            return
        for batch in tqdm(datamodule.train_dataloader(), desc=f"Fitting property scalers"):
            for property_name in property_names:
                # concat all values in train dataset for this given property
                property_values[property_name].append(batch[property_name])

        for property_name in property_names:
            property_embeddings[property_name].fit_scaler(
                all_data=maybe_to_tensor(values=property_values[property_name])
            )

    def on_fit_start(self, trainer: pl.Trainer, pl_module: DiffusionLightningModule):
        model: GemNetTDenoiser = pl_module.diffusion_module.model

        # model.property_embeddings: torch.nn.ModuleDict always exists
        self._compute_property_scalers(
            datamodule=trainer.datamodule, property_embeddings=model.property_embeddings
        )

        if hasattr(model, "property_embeddings_adapt"):
            # this is a fine tune model
            self._compute_property_scalers(
                datamodule=trainer.datamodule, property_embeddings=model.property_embeddings_adapt
            )
