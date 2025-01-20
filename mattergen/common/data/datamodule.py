# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import CrystalDataset


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class CrystDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: CrystalDataset,
        num_workers: DictConfig,
        batch_size: DictConfig,
        val_dataset: CrystalDataset | None = None,
        test_dataset: CrystalDataset | None = None,
        **_,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.datasets = [train_dataset, val_dataset, test_dataset]

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )

    def val_dataloader(self, shuffle: bool = False) -> DataLoader | None:
        return (
            DataLoader(
                self.val_dataset,
                shuffle=shuffle,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
                collate_fn=collate,
            )
            if self.val_dataset is not None
            else None
        )

    def test_dataloader(self, shuffle: bool = False) -> DataLoader | None:
        return (
            DataLoader(
                self.test_dataset,
                shuffle=shuffle,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
                collate_fn=collate,
            )
            if self.test_dataset is not None
            else None
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )
