"""
File: src/data_converters/word_role_datamodule.py
Creation Date 2024-06-08

This file contains LightningDataModule definitions for actually preprocessing, loading, and
transforming data for use in PyTorch Lightning. A torch Dataset object is used in conjunction.
Preprocessing our data also includes converting from raw XML GZ files to the correct format.
"""
import os
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import gdown

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from roles import *
from convert_data import build_word_vocabulary, convert_files

log = logging.getLogger('main.word_role_datamodule')

class WordRoleDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return idx


class WordRoleDataModule(LightningDataModule):
    def __init__(self, data_dir, role_set: Roles, corpus_filepath: str, train_frac=0.005, val_files=(1, 2, 3), test_files=(4, 5, 6), num_words=50000,
                 num_workers=2, batch_size=32, pin_memory=True):
        super().__init__()
        self.save_hyperparameters()
        self.xml_gz_dir = os.path.join(data_dir, 'xml_gz')
        self.dir_names = ['train', 'val', 'test']
        for dir_ in self.dir_names:
            os.makedirs(os.path.join(self.xml_gz_dir, dir_), exist_ok=True)
        # Read the corpus, which contains the original download links to the XML GZ
        self.corpus = pd.read_csv(corpus_filepath, index_col='num')
        self.train_files: list = []
        self.word_vocab: dict = {}
        self.role_vocab: dict = {}

    def prepare_data(self) -> None:
        """
        Here, we download the files we need, process them,
        and place them into a different directory.
        We also run the sample duplication in this step as well,
        but because the output is all numbers, we can save a lot of
        space by saving in uint16 dtype (max value = 65355).
        :return:
        """
        # First, we have to generate the list of train files.
        # We are given a fraction and the set in stone list of
        # validation and testing. Remove these from the corpus dataframe,
        # and sample it afterwards...
        train_universe = self.corpus.drop(index=self.hparams.val_files + self.hparams.test_files)
        self.train_files = train_universe.sample(frac=self.hparams.train_frac).index.tolist()
        log.info(f'Train fraction - {int(self.hparams.train_frac * 100)}% = {len(self.train_files)} training files')
        log.info(f'Train files - {self.train_files}')

        # Download the XML GZ files we need
        log.info('Downloading XML GZ files...')
        for data_name, fileset in zip(self.dir_names, [self.train_files, self.hparams.val_files, self.hparams.test_files]):
            for file_num in tqdm(fileset, desc=data_name):
                url, name = self.corpus.loc[file_num, 'url'], self.corpus.loc[file_num, 'name']
                filepath = os.path.join(self.xml_gz_dir, data_name, name)
                if not os.path.exists(filepath):
                    gdown.download(url=url, output=filepath, quiet=True)

        # Build the vocabulary!
        self.word_vocab = build_word_vocabulary(self.train_files, num_words=self.hparams.num_words)

    def setup(self, stage=None):
        # The sample duplication is complete, and here we simply create
        # the correct dataset objects. Since the dataset has already been
        # split up by training, validation, and testing, we can simply read them.
        return

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return
