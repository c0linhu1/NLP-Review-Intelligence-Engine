"""
preprocessing.py

raw test contains html tags, weird unicode, incosistensies, whitespace ,etc
first step is always cleaning/standardizing text into a clean format that 
every downstream component can rely on.

"""

import re
import html
import unicodedata
from pathlib import Path

import pandas as pd
import spacy
from datasets import load_dataset

def load_and_clean_data(n_samples = 50000, random_state = 42, save = True):
    """
    Load Amazon Polarity dataset: sample clean - save to parquet

    Parquet is a columnar file format - faster to read/write than CSV and
    preserves data types (CSVs lose list columns), and compresses well.
    Once we save to parquet, every future run loads in <1 second instead of
    re-downloading the full dataset from HuggingFace
    """

    # caching - check to see if exists so we dont need to download everytime
    cache_path = Path("data/reviews_clean.parquet")
    if cache_path.exists():
        print('Loading cached data from {cache_path}')
        return pd.read_parquet(cache_path)

    # load from HuggingFace 
    print('Loading Amazon Polarity dataset')
    dataset = load_dataset('amazon_polarity', split = 'test')

    df = pd.DataFrame(dataset).sample(
        n = n_samples, random_state = random_state).reset_index(drop = True)
