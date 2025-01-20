# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:

    # This is for CLI applications that need to reuse a CLI parameter in multiple places
    # in the config file. The idea is that you use `my_cli params.output_dir=foobar`
    # and in other places in the config file `output_dir: ${params.output_dir}`
    params: dict[str, Any] = field(default_factory=dict)

    checkpoint_path: str | None = None  # Required if train == False

    # if load_original is True then we load original weights in validation mode instead of EMA
    load_original: bool = False

    # When auto_resume is set to `True` the trainer saves a copy of each checkpoint in
    # {trainer.default_root_dir}/checkpoints. Before starting training, we look in this
    # directory for a checkpoint from which to resume training.
    auto_resume: bool = False

    # DiffusionLightningModule
    lightning_module: dict[str, Any] = field(default_factory=dict)

    # pytorch_lightning.Trainer
    trainer: dict[str, Any] = field(default_factory=dict)

    # LightningDataModule
    data_module: dict[str, Any] = field(default_factory=dict)
