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