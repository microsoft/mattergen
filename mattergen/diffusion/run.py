# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import logging
import os
import random
import re
from glob import glob
from typing import Any, Mapping, TypeVar

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import SaveConfigCallback
from pytorch_lightning.utilities import rank_zero_only

from mattergen.common.utils.config_utils import get_config
from mattergen.diffusion.config import Config
from mattergen.diffusion.exceptions import AmbiguousConfig
from mattergen.diffusion.lightning_module import DiffusionLightningModule

T = TypeVar("T")

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def maybe_instantiate(instance_or_config: T | Mapping, expected_type=None, **kwargs) -> T:
    """
    If instance_or_config is a mapping with a _target_ field, instantiate it.
    Otherwise, return it as is.
    """
    if isinstance(instance_or_config, Mapping) and "_target_" in instance_or_config:
        instance = instantiate(instance_or_config, **kwargs)
    else:
        instance = instance_or_config
    assert expected_type is None or isinstance(
        instance, expected_type
    ), f"Expected {expected_type}, got {type(instance)}"
    return instance


def _find_latest_checkpoint(dirpath: str) -> str | None:
    """Finds the most recent checkpoint inside `dirpath`."""

    # checkpoint names are like "epoch=0-step=0.ckpt."
    # Find the checkpoint with highest epoch:
    def extract_epoch(ckpt):
        match = re.search(r"epoch=(\d+)", ckpt)
        if match:
            return int(match.group(1))
        return -1

    ckpts = glob(f"{dirpath}/*.ckpt")
    epochs = np.array([extract_epoch(ckpt) for ckpt in ckpts])
    if len(epochs) == 0 or epochs.max() < 0:
        # No checkpoints found.
        return None
    latest_checkpoint = ckpts[epochs.argmax()]
    return latest_checkpoint


class SimpleParser:
    def save(self, config, path, **_):
        with open(path, "w") as f:
            yaml.dump(config, f)


class AddConfigCallback(Callback):
    """Adds a copy of the config to the checkpoint, so that `load_from_checkpoint` can use it to instantiate everything."""

    def __init__(self, config: dict[str, Any]):
        self._config_dict = config

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: dict[str, Any]
    ) -> None:
        checkpoint["config"] = self._config_dict


def main(
    config: Config | DictConfig, save_config: bool = True, seed: int | None = None
) -> tuple[pl.Trainer, pl.LightningModule]:
    """
    Main entry point to train and evaluate a diffusion model.

    save_config: if True, the config will be saved both as a YAML file and in each checkpoint. This doesn't work if the config contains things that can't be `yaml.dump`-ed, so
    if you don't care about saving and loading checkpoints and want to use a config that contains things like `torch.nn.Module`s already instantiated, set this to False.
    """
    if config.checkpoint_path and config.auto_resume:
        raise AmbiguousConfig(
            f"Ambiguous config: you set both a checkpoint path {config.checkpoint_path} and `auto_resume` which means automatically select a checkpoint path to resume from."
        )

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    trainer: pl.Trainer = maybe_instantiate(config.trainer, pl.Trainer)

    if save_config:
        if isinstance(config, DictConfig):
            config_as_dict = OmegaConf.to_container(config, resolve=True)
            # This callback will save a config.yaml file.
            trainer.callbacks.append(
                SaveConfigCallback(
                    parser=SimpleParser(),
                    config=config_as_dict,
                    overwrite=True if config.auto_resume else False,
                )
            )

            # This callback will add a copy of the config to each checkpoint.
            trainer.callbacks.append(AddConfigCallback(config_as_dict))
        else:
            raise NotImplementedError
    datamodule: pl.LightningDataModule = maybe_instantiate(
        config.data_module, pl.LightningDataModule
    )

    # If checkpoint_path is provided training will be resumed from this point.
    # Beware: the old checkpoint will be deleted when a new one is saved.

    ckpt_path = config.checkpoint_path
    if config.auto_resume:
        # Add an additional checkpointer with a fixed directory path to restore from.
        dirpath = os.path.join(trainer.default_root_dir, "checkpoints")
        trainer.callbacks.append(ModelCheckpoint(dirpath=dirpath))
        ckpt_path = _find_latest_checkpoint(dirpath)
    pl_module: DiffusionLightningModule = maybe_instantiate(
        config.lightning_module, DiffusionLightningModule
    )
    if rank_zero_only.rank == 0 and isinstance(trainer.logger, pl.loggers.WandbLogger):
        # Log the config to wandb so that it shows up in the portal.
        trainer.logger.experiment.config.update(
            {**OmegaConf.to_container(config, resolve=True)},
            allow_val_change=True,
        )
    trainer.fit(
        pl_module,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )

    return trainer, pl_module


def cli(argv: list[str] | None) -> None:
    """
    Args:
        argv: list of command-line arguments as strings, or None. If None,
          command-line arguments will be got from sys.argv
    """

    parser = argparse.ArgumentParser(allow_abbrev=False)  # prevent prefix matching issues
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to use. If not provided, a random seed will be used.",
    )
    args, argv = parser.parse_known_args(argv)

    # Create config from command-line arguments.
    config = get_config(argv, Config)
    main(config, seed=args.seed)


if __name__ == "__main__":
    cli(argv=None)
