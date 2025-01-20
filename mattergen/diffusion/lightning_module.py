# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Generic, Optional, Protocol, Sequence, TypeVar, Union

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW, Optimizer
from tqdm import tqdm

from mattergen.diffusion.config import Config
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.diffusion_module import DiffusionModule

T = TypeVar("T", bound=BatchedData)


class OptimizerPartial(Protocol):
    """Callable to instantiate an optimizer."""

    def __call__(self, params: Any) -> Optimizer:
        raise NotImplementedError


class SchedulerPartial(Protocol):
    """Callable to instantiate a learning rate scheduler."""

    def __call__(self, optimizer: Optimizer) -> Any:
        raise NotImplementedError


def get_default_optimizer(params):
    return AdamW(params=params, lr=1e-4, weight_decay=0, amsgrad=True)


class DiffusionLightningModule(pl.LightningModule, Generic[T]):
    """LightningModule for instantiating and training a DiffusionModule."""

    def __init__(
        self,
        diffusion_module: DiffusionModule[T],
        optimizer_partial: Optional[OptimizerPartial] = None,
        scheduler_partials: Optional[Sequence[Dict[str, Union[Any, SchedulerPartial]]]] = None,
    ):
        """_summary_

        Args:
            diffusion_module: The diffusion module to use.
            optimizer_partial: Used to instantiate optimizer.
            scheduler_partials: used to instantiate learning rate schedulers
        """
        super().__init__()
        scheduler_partials = scheduler_partials or []
        optimizer_partial = optimizer_partial or get_default_optimizer
        self.save_hyperparameters(
            ignore=("optimizer_partial", "scheduler_partials", "diffusion_module")
        )

        self.diffusion_module = diffusion_module
        self._optimizer_partial = optimizer_partial
        self._scheduler_partials = scheduler_partials

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: Optional[str] = None,
        **kwargs,
    ) -> DiffusionLightningModule:
        """Load model from checkpoint. kwargs are passed to hydra's instantiate and can override
        arguments from the checkpoint config."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # The config should have been saved in the checkpoint by AddConfigCallback in run.py
        config = Config(**checkpoint["config"])
        try:
            lightning_module = instantiate(config.lightning_module, **kwargs)
        except InstantiationException as e:
            print("Could not instantiate model from the checkpoint.")
            print(
                "If the error is due to an unexpected argument because the checkpoint and the code have diverged, try using load_from_checkpoint_and_config instead."
            )
            raise e
        assert isinstance(lightning_module, cls)

        # Restore state of the DiffusionLightningModule.
        lightning_module.load_state_dict(checkpoint["state_dict"])
        return lightning_module

    @classmethod
    def load_from_checkpoint_and_config(
        cls,
        checkpoint_path: str,
        config: DictConfig,
        map_location: Optional[str] = None,
        strict: bool = True,
    ) -> tuple[DiffusionLightningModule, torch.nn.modules.module._IncompatibleKeys]:
        """Load model from checkpoint, but instead of using the config stored in the checkpoint,
        use the config passed in as an argument. This is useful when, e.g., an unused argument was
        removed in the code but is still present in the checkpoint config."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        lightning_module = instantiate(config)
        assert isinstance(lightning_module, cls)

        # Restore state of the DiffusionLightningModule.
        result = lightning_module.load_state_dict(checkpoint["state_dict"], strict=strict)

        return lightning_module, result

    def configure_optimizers(self) -> Any:
        optimizer = self._optimizer_partial(params=self.diffusion_module.parameters())
        if self._scheduler_partials:
            lr_schedulers = [
                {
                    **scheduler_dict,
                    "scheduler": scheduler_dict["scheduler"](
                        optimizer=optimizer,
                    ),
                }
                for scheduler_dict in self._scheduler_partials
            ]

            return [
                optimizer,
            ], lr_schedulers
        else:
            return optimizer

    def training_step(self, train_batch: T, batch_idx: int) -> STEP_OUTPUT:
        return self._calc_loss(train_batch, True)

    def validation_step(self, val_batch: T, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._calc_loss(val_batch, False)

    def test_step(self, test_batch: T, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._calc_loss(test_batch, False)

    def _calc_loss(self, batch: T, train: bool) -> Optional[STEP_OUTPUT]:
        """Calculate loss and metrics given a batch of clean data."""
        loss, metrics = self.diffusion_module.calc_loss(batch)
        # Log the results
        step_type = "train" if train else "val"
        batch_size = batch.get_batch_size()
        self.log(
            f"loss_{step_type}",
            loss,
            on_step=train,
            on_epoch=True,
            prog_bar=train,
            batch_size=batch_size,
            sync_dist=True,
        )
        for k, v in metrics.items():
            self.log(
                f"{k}_{step_type}",
                v,
                on_step=train,
                on_epoch=True,
                prog_bar=train,
                batch_size=batch_size,
                sync_dist=True,
            )
        return loss
