# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import fnmatch
import os
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np
from huggingface_hub import hf_hub_download
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

PRETRAINED_MODEL_NAME = Literal[
    "mattergen_base",
    "chemical_system",
    "space_group",
    "dft_mag_density",
    "dft_band_gap",
    "ml_bulk_modulus",
    "dft_mag_density_hhi_score",
    "chemical_system_energy_above_hull",
]


def find_local_files(local_path: str, glob: str = "*", relative: bool = False) -> list[str]:
    """
    Find files in the given directory or blob storage path, and return the list of files
    matching the given glob pattern. If relative is True, the returned paths are relative
    to the given directory or blob storage path.

    Args:
        blob_or_local_path: path to the directory or blob storage path
        glob: glob pattern to match. By default, all files are returned.
        relative: whether to return relative paths. By default, absolute paths are returned.

    Returns:
        list of paths to files matching the given glob pattern.
    """
    # list all files here, filtering happens in the `fnmatch.filter` step
    local_files = [x for x in Path(local_path).rglob("*") if os.path.isfile(x)]
    files_list = [str(x.relative_to(local_path)) if relative else str(x) for x in local_files]
    return fnmatch.filter(files_list, glob)


@dataclass(frozen=True)
class MatterGenCheckpointInfo:
    model_path: str
    load_epoch: int | Literal["best", "last"] | None = "last"
    config_overrides: list[str] = field(default_factory=list)
    split: str = "val"
    strict_checkpoint_loading: bool = True

    @classmethod
    def from_hf_hub(
        cls,
        model_name: PRETRAINED_MODEL_NAME,
        repository_name: str = "microsoft/mattergen",
        config_overrides: list[str] = None,
    ):
        """
        Instantiate a MatterGenCheckpointInfo object from a model hosted on the Hugging Face Hub.

        """
        
        # change endpoint to hf-mirror.com
        hf_mirror_url = "https://hf-mirror.com"

        # add endpoint to use the mirror
        hf_hub_download(
            repo_id=repository_name,
            filename=f"checkpoints/{model_name}/checkpoints/last.ckpt",
            endpoint=hf_mirror_url
        )

        # add endpoint to use the mirror
        config_path = hf_hub_download(
            repo_id=repository_name,
            filename=f"checkpoints/{model_name}/config.yaml",
            endpoint=hf_mirror_url
        )
        return cls(
            model_path=Path(config_path).parent,
            config_overrides=config_overrides or [],
            load_epoch="last",
        )

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model_path"] = str(self.model_path)  # we cannot put Path object in mongo DB
        return d

    @classmethod
    def from_dict(cls, d) -> "MatterGenCheckpointInfo":
        d = d.copy()
        d["model_path"] = Path(d["model_path"])
        # no longer used
        if "load_data" in d:
            del d["load_data"]
        return cls(**d)

    @property
    def config(self) -> DictConfig:
        with initialize_config_dir(str(self.model_path)):
            cfg = compose(config_name="config", overrides=self.config_overrides)
            return cfg

    @cached_property
    def checkpoint_path(self) -> str:
        """
        Search for checkpoint files in the given directory, and return the path
        to the checkpoint with the given epoch number or the best checkpoint if load_epoch is "best".
        "Best" is selected via the lowest validation loss, which is stored in the checkpoint filename.
        Assumes that the checkpoint filenames are of the form "epoch=1-val_loss=0.1234.ckpt" or 'last.ckpt'.

        Returns:
            Path to the checkpoint file to load.
        """
        # look for checkpoints recursively in the given directory or blob storage path.
        # I.e., if the path is '/path/', we will find .ckpt files in '/path/version_0/checkpoints'
        # and '/path/version_1/checkpoints', and so on.
        model_path = str(self.model_path)
        ckpts = find_local_files(local_path=model_path, glob="*.ckpt")
        assert len(ckpts) > 0, f"No checkpoints found at {model_path}"
        if self.load_epoch == "last":
            assert any(
                [x.endswith("last.ckpt") for x in ckpts]
            ), "No last.ckpt found in checkpoints."
            return [x for x in ckpts if x.endswith("last.ckpt")][0]
        # Drop last.ckpt to exclude it from the epoch selection
        ckpts = [x for x in ckpts if not x.endswith("last.ckpt")]

        # Convert strings to Path to be able to use the .parts attribute
        ckpt_paths = [Path(x) for x in ckpts]
        # Extract the epoch number and validation loss from the checkpoint filenames
        ckpt_epochs = np.array(
            [
                int(ckpt.parts[-1].split(".ckpt")[0].split("-")[0].split("=")[1])
                for ckpt in ckpt_paths
            ]
        )
        ckpt_val_losses = np.array(
            [
                (
                    float(ckpt.parts[-1].replace(".ckpt", "").split("-")[1].split("=")[1])
                    if "loss_val" in ckpt.parts[-1]
                    else 99999999.9
                )
                for ckpt in ckpt_paths
            ]
        )

        # Determine the matching checkpoint index.
        if self.load_epoch == "best":
            ckpt_ix = ckpt_val_losses.argmin()
        elif isinstance(self.load_epoch, int):
            assert (
                self.load_epoch in ckpt_epochs
            ), f"Epoch {self.load_epoch} not found in checkpoints."
            ckpt_ix = (ckpt_epochs == self.load_epoch).nonzero()[0][0].item()
        else:
            raise ValueError(f"Unrecognized load_epoch {self.load_epoch}")
        ckpt = ckpts[ckpt_ix]
        return ckpt
