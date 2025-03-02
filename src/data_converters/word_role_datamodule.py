"""
File: src/data_converters/word_role_datamodule.py
Creation Date 2024-06-08

This file contains LightningDataModule definitions for actually preprocessing, loading, and
transforming data for use in PyTorch Lightning. A torch Dataset object is used in conjunction.
Preprocessing our data also includes converting from raw XML GZ files to the correct format.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import gdown
import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from src.data_converters.convert_data import build_word_vocabulary, convert_files
from src.data_converters.roles import *


log = logging.getLogger('main.word_role_datamodule')

class WordRoleDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return idx


class WordRoleDataModule(LightningDataModule):
    def __init__(self, data_dir, role_set: Roles, corpus_filepath: str, train_frac=0.005, val_files=(1, 2, 3),
                 test_files=(4, 5, 6), num_words=50000, num_workers=2, batch_size=32, pin_memory=True):
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
        # Set the unknown and missing role IDs
        self.unk_word_id = num_words
        self.missing_word_id = num_words + 1
        self.unk_role_id = len(self.hparams.role_set.ROLE_SET)
        self.missing_role_id = len(self.hparams.role_set.ROLE_SET) + 1

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
        # We are given a fraction of how many we need. This is taken from the
        # list of all files AFTER the validation and test files have been removed.
        # To keep it consistent with the previous version, we use a stratified sample.
        # The stratification is rounded from the provided fraction, and we select the FIRST
        # non-validation and non-test file, and return to normal gap e.g. if we are choosing every 10 files,
        # and file #11 and file #12 are part of validation and testing, then the chosen training files will
        # be #1, #13 (first valid training file), #21 (return to normal gap), #31, etc.
        # THIS IS PURELY TO KEEP IT CONSISTENT WITH PREVIOUS CODEBASE.
        # train_universe = self.corpus.drop(index=self.hparams.val_files + self.hparams.test_files)
        # self.train_files = train_universe.sample(frac=self.hparams.train_frac).index.tolist()
        num_train_files = min(int(round(self.corpus.shape[0] * self.hparams.train_frac)),
                              self.corpus.shape[0] - len(self.hparams.val_files) - len(self.hparams.test_files))
        log.info(f'Number of training files: {num_train_files}')
        # Calculate the step
        step = int(1.0 / self.hparams.train_frac)
        self.train_files = self.corpus.index.values[::step]
        # Find the locations where we have selected a validation or test file
        locs = np.isin(self.train_files, self.hparams.val_files) | np.isin(self.train_files, self.hparams.test_files)
        # While we don't have any of these locations, add 1 to the spots where we have chosen them,
        # and recalculate the locations.
        while np.any(locs):
            self.train_files[locs] += 1
            locs = (np.isin(self.train_files, self.hparams.val_files) |
                    np.isin(self.train_files, self.hparams.test_files))
        # Cut it down to however we need...
        self.train_files = self.train_files[:num_train_files]

        # Download the XML GZ files we need
        log.info('Downloading XML GZ files...')
        for data_name, fileset in zip(self.dir_names, [self.train_files, self.hparams.val_files, self.hparams.test_files]):
            for file_num in tqdm(fileset, desc=data_name):
                url, name = self.corpus.loc[file_num, 'url'], self.corpus.loc[file_num, 'name']
                filepath = os.path.join(self.xml_gz_dir, data_name, name)
                if not os.path.exists(filepath):
                    gdown.download(url=url, output=filepath, quiet=True)
        log.info(f'Train files: {self.train_files}')

        # Build the vocabulary using the files that were just downloaded into
        # the training directory (use glob for this).
        self.word_vocab = build_word_vocabulary(glob.glob(os.path.join(self.xml_gz_dir, 'train', '*.xml.gz')), num_words=self.hparams.num_words)
        self.role_vocab = self.hparams.role_set.ROLE_SET
        log.info(f'Role Vocabulary\n{self.role_vocab}')

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
