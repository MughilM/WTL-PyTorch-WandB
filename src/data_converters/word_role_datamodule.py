"""
File: src/data_converters/word_role_datamodule.py
Creation Date 2024-06-08

This file contains LightningDataModule definitions for actually preprocessing, loading, and
transforming data for use in PyTorch Lightning. A torch Dataset object is used in conjunction.
Preprocessing our data also includes converting from raw XML GZ files to the correct format.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


class WordRoleDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return idx


class WordRoleDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self) -> None:
        return

    def setup(self, stage=None):
        return

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return
