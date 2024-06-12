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
