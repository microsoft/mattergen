# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.cli import SaveConfigCallback

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT, get_device
from mattergen.diffusion.run import AddConfigCallback, SimpleParser, maybe_instantiate

logger = logging.getLogger(__name__)


def init_adapter_lightningmodule_from_pretrained(
    adapter_cfg: DictConfig, lightning_module_cfg: DictConfig
) -> Tuple[pl.LightningModule, DictConfig]:

    if adapter_cfg.model_path is not None:
        if adapter_cfg.pretrained_name is not None:
            logger.warning(
                "pretrained_name is provided, but will be ignored since model_path is also provided."
            )
        model_path = Path(hydra.utils.to_absolute_path(adapter_cfg.model_path))
        ckpt_info = MatterGenCheckpointInfo(model_path, adapter_cfg.load_epoch)
    elif adapter_cfg.pretrained_name is not None:
        assert (
            adapter_cfg.model_path is None
        ), "model_path must be None when pretrained_name is provided."
        ckpt_info = MatterGenCheckpointInfo.from_hf_hub(adapter_cfg.pretrained_name)

    ckpt_path = ckpt_info.checkpoint_path

    version_root_path = Path(ckpt_path).relative_to(ckpt_info.model_path).parents[1]
    config_path = ckpt_info.model_path / version_root_path

    # load pretrained model config.
    if (config_path / "config.yaml").exists():
        pretrained_cfg_path = config_path
    else:
        pretrained_cfg_path = config_path.parent.parent

    # global hydra already initialized with @hydra.main
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    with hydra.initialize_config_dir(str(pretrained_cfg_path.absolute()), version_base="1.1"):
        pretrained_cfg = hydra.compose(config_name="config")

    # compose adapter lightning_module config.

    ## copy denoiser config from pretrained model to adapter config.
    diffusion_module_cfg = deepcopy(pretrained_cfg.lightning_module.diffusion_module)
    denoiser_cfg = diffusion_module_cfg.model

    with open_dict(adapter_cfg.adapter):
        for k, v in denoiser_cfg.items():
            # only legacy denoiser configs should contain property_embeddings_adapt
            if k != "_target_" and k != "property_embeddings_adapt":
                adapter_cfg.adapter[k] = v

            # do not adapt an existing <property_embeddings> field.
            if k == "property_embeddings":
                for field in v:
                    if field in adapter_cfg.adapter.property_embeddings_adapt:
                        adapter_cfg.adapter.property_embeddings_adapt.remove(field)

        # replace original GemNetT model with GemNetTCtrl model.
        adapter_cfg.adapter.gemnet["_target_"] = "mattergen.common.gemnet.gemnet_ctrl.GemNetTCtrl"

        # GemNetTCtrl model has additional input parameter condition_on_adapt, which needs to be set via property_embeddings_adapt.
        adapter_cfg.adapter.gemnet.condition_on_adapt = list(
            adapter_cfg.adapter.property_embeddings_adapt
        )

    # copy adapter config back into diffusion module config
    with open_dict(diffusion_module_cfg):
        diffusion_module_cfg.model = adapter_cfg.adapter
    with open_dict(lightning_module_cfg):
        lightning_module_cfg.diffusion_module = diffusion_module_cfg

    lightning_module = hydra.utils.instantiate(lightning_module_cfg)

    ckpt: dict = torch.load(ckpt_path, map_location=get_device())
    pretrained_dict: OrderedDict = ckpt["state_dict"]
    scratch_dict: OrderedDict = lightning_module.state_dict()
    scratch_dict.update(
        (k, pretrained_dict[k]) for k in scratch_dict.keys() & pretrained_dict.keys()
    )
    lightning_module.load_state_dict(scratch_dict, strict=True)

    # freeze pretrained weights if not full finetuning.
    if not adapter_cfg.full_finetuning:
        for name, param in lightning_module.named_parameters():
            if name in set(pretrained_dict.keys()):
                param.requires_grad_(False)

    return lightning_module, lightning_module_cfg


@hydra.main(
    config_path=str(MODELS_PROJECT_ROOT / "conf"), config_name="finetune", version_base="1.1"
)
def mattergen_finetune(cfg: omegaconf.DictConfig):
    # Tensor Core acceleration (leads to ~2x speed-up during training)
    torch.set_float32_matmul_precision("high")
    trainer: pl.Trainer = maybe_instantiate(cfg.trainer, pl.Trainer)
    datamodule: pl.LightningDataModule = maybe_instantiate(cfg.data_module, pl.LightningDataModule)

    # establish an adapter model
    pl_module, lightning_module_cfg = init_adapter_lightningmodule_from_pretrained(
        cfg.adapter, cfg.lightning_module
    )

    # replace denoiser config with adapter config.
    with open_dict(cfg):
        cfg.lightning_module = lightning_module_cfg

    config_as_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(config_as_dict, indent=4))
    # This callback will save a config.yaml file.
    trainer.callbacks.append(
        SaveConfigCallback(
            parser=SimpleParser(),
            config=config_as_dict,
            overwrite=True,
        )
    )
    # This callback will add a copy of the config to each checkpoint.
    trainer.callbacks.append(AddConfigCallback(config_as_dict))

    trainer.fit(
        model=pl_module,
        datamodule=datamodule,
        ckpt_path=None,
    )


if __name__ == "__main__":
    mattergen_finetune()
