"""
File: src/main.py
Author: Mughil Pari
Creation Date: 2024-06-03

The entry point file for all the functionality of the codebase.
It handles training, evaluating, and converting the data files from the
original XML GZ to simple text files for ingestion.
Configuration for these operations are handled by Hydra, and are housed in
the /config folder. Model training and outputs are performed using PyTorch Lightning
and then uploaded to Weights and Biases for automated metric comparisons.
Metrics include both raw accuracy and thematic fit evaluation task metrics.
"""
import pyrootutils
# Set up the root of the entire projects. This avoids weird import errors when running the program
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=['.git', '.idea'],
    pythonpath=True
)

import os
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer


torch.set_float32_matmul_precision('medium')
os.environ['HYDRA_FULL_ERROR'] = '1'


def train(cfg: DictConfig) -> None:
    logger.info(f'Full Hydra configuration:\n{OmegaConf.to_yaml(cfg)}')
    # Set the seed for everything (pytorch, numpy, etc.)
    if cfg.get('seed'):
        pl.seed_everything(cfg.seed, workers=True)
    # Create the data directory in case it's missing
    os.makedirs(cfg.paths.data_dir, exist_ok=True)


@hydra.main(version_base='1.3', config_path='../config', config_name='train.yaml')
def main(cfg: DictConfig) -> None:
    train(cfg)
    return


if __name__ == '__main__':
    logger = logging.getLogger('train')
    main()


