"""
File: src/data_converters/convert_data.py
Creation Date: 2024-06-08

This file contains all the logic to convert the raw XML files into jsonline files that
will be ingested downstream. The raw files will be downloaded one by one and then deleted
in order to save space. However, depending on the number of threads, up to 8 or 16 of these files
can be downloaded and read from at the same time. The end result will be a single jsonline file for
each of training, validation, and testing, as well as a few description files which describes the data
(files used, role and word vocabulary, etc.). The LightningDataModule will take care of actually
duplicating these samples and saving them.
"""
import os
import numpy as np
import xml.etree.ElementTree as ET
import jsonlines
import gzip
import time

import roles
role_set = roles.Roles2Args3Mods()

VAL_FILES = [1, 2, 3]
TEST_FILES = [4, 5, 6]
TRAIN_FRAC = [7, 8, 9]
NUM_WORDS = 50000

def build_word_vocabulary(train_files, num_words=NUM_WORDS) -> dict:
    """
    Before we actually run through the head words, we need to build the vocabulary.
    Here, we take all words that are present in the lemmas (instead of just the head words).
    This can also be parallelizable if we need to.
    It is assumed that the files are already downloaded to the correct location.
    :param num_words:
    :return: A keyed dictionary which maps words to their corresponding integer keys
    """


def convert_files(files, word_vocab, role_vocab, num_workers=2):
    """
    Function to batch convert a large list of files. The num_workers argument
    controls how many threads to use for the conversion. For best performance,
    use the number of CPU cores on the machine.
    :param files: The list of files to convert
    :param word_vocab: The word vocabulary to use when coding to integers
    :param role_vocab: The role vocabulary
    :param num_workers: The number of threads to spool up
    :return:
    """


