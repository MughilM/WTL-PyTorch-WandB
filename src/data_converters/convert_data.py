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
from collections import Counter
import numpy as np
import xml.etree.ElementTree as ET
import jsonlines
import gzip
import time
from tqdm.contrib.concurrent import thread_map

import src.data_converters.roles
role_set = src.data_converters.roles.Roles2Args3Mods()

def build_word_vocabulary(files, num_words) -> dict:
    """
    Before we actually run through the head words, we need to build the vocabulary.
    Here, we take all words that are present in the lemmas (instead of just the head words).
    This can also be parallelizable if we need to.
    It is assumed that the files are already downloaded to the correct location.
    :param files: List of XML GZIP files. These are extracted and read in memory.
    :param num_words: Any words that aren't in the top num_words in terms of frequency get tossed out.
    :return: A keyed dictionary which maps words to their corresponding integer keys
    """
    # Create a subfunction that builds the full vocabulary for one file
    def build_file_vocab(gzip_file):
        # gzip_file is assumed to be .xml.gz file.
        xml = ET.fromstring(gzip.GzipFile(gzip_file).read())
        # For each s/Frame/Arg argument, we take all the lemmas (it's possible there's more thon one,
        # and keep track of a counter)
        c = Counter()
        for arg_node in xml.findall('s/Frame/Arg'):
            c.update(arg_node.attrib.get('lemmas').strip().split())
        return c
    # This is the master counter. For every file we analyze, we update the
    # master counter with the frequencies.
    master_counter = Counter()
    file_counter_objs = thread_map(build_file_vocab, files, desc='Building vocab', chunksize=10000,
                                   max_workers=8)
    for file_counter in file_counter_objs:
        master_counter.update(file_counter)  # Update the master
    # Get the most common num_words from the frequency list, and 0-index them
    # in order of most common to least common.
    # Ex. if num_words = 50000, then most common word is 0, while 50000th common is 49999.
    master_counter = {word: index for index, (word, count) in enumerate(master_counter.most_common(num_words))}
    return master_counter



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



