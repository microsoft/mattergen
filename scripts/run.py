# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging

import hydra
import omegaconf
import torch
from omegaconf import OmegaConf

from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from mattergen.diffusion.config import Config
from mattergen.diffusion.run import main

logger = logging.getLogger(__name__)


@hydra.main(
    config_path=str(MODELS_PROJECT_ROOT / "conf"), config_name="default", version_base="1.1"
)
def mattergen_main(cfg: omegaconf.DictConfig):
    # Tensor Core acceleration (leads to ~2x speed-up during training)
    torch.set_float32_matmul_precision("high")
    # Make merged config options
    # CLI options take priority over YAML file options
    schema = OmegaConf.structured(Config)
    config = OmegaConf.merge(schema, cfg)
    OmegaConf.set_readonly(config, True)  # should not be written to
    print(OmegaConf.to_yaml(cfg, resolve=True))

    main(config)


if __name__ == "__main__":
    mattergen_main()
