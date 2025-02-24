# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../scripts"))

import hydra
import pytest

from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from mattergen.scripts.run import mattergen_main

CONFIG_DIR = os.path.join(MODELS_PROJECT_ROOT, "conf")


@pytest.mark.parametrize("config_name", ["default"])
def test_train_on_one_batch(config_name: str) -> None:
    # Tests that the model can be instantiated and trained on one batch.

    # override some config options to make the run short.
    overrides = [
        "trainer.max_epochs=1",
        "+trainer.overfit_batches=1",
        "trainer.check_val_every_n_epoch=1",
        "lightning_module.diffusion_module.model.gemnet.num_blocks=1",
        "lightning_module.diffusion_module.model.gemnet.max_neighbors=5",
        "lightning_module.diffusion_module.model.hidden_dim=16",
        "data_module.batch_size.train=8",
        "data_module.batch_size.val=8",
        "data_module.batch_size.test=8",
        "+trainer.limit_val_batches=1",
        "+trainer.limit_test_batches=1",
        "trainer.accelerator=cpu",
        "~trainer.logger",  # wandb does not work in CI
    ]
    with hydra.initialize_config_dir(config_dir=CONFIG_DIR):
        config = hydra.compose(config_name=config_name, overrides=overrides)

    _ = mattergen_main(config)
    # if we reach this, the test passed.
    assert True
