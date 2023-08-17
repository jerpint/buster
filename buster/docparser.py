import glob
import logging
import os
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from openai.embeddings_utils import get_embedding
from tqdm import tqdm

from buster.parser import HuggingfaceParser, Parser, SphinxParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

supported_docs = {
    "mila": {
        "base_url": "https://docs.mila.quebec/",
        "filename": "documents_mila.csv",
        "parser": SphinxParser,
    },
    "orion": {
        "base_url": "https://orion.readthedocs.io/en/stable/",
        "filename": "documents_orion.csv",
        "parser": SphinxParser,
    },
    "pytorch": {
        "base_url": "https://pytorch.org/docs/stable/",
        "filename": "documents_pytorch.csv",
        "parser": SphinxParser,
    },
    "huggingface": {
        "base_url": "https://huggingface.co/docs/transformers/",
        "filename": "documents_huggingface.csv",
        "parser": HuggingfaceParser,
    },
    "lightning": {
        "base_url": "https://pytorch-lightning.readthedocs.io/en/stable/",
        "filename": "documents_lightning.csv",
        "parser": SphinxParser,
    },
    "godot": {
        "base_url": "https://docs.godotengine.org/en/stable/",
        "filename": "documents_godot.csv",
        "parser": SphinxParser,
    },
}


def get_document(
    root_dir: str,
    file: str,
    base_url: str,
    parser_cls: Type[Parser],
    min_section_length: int = 100,
    max_section_length: int = 2000,
) -> pd.DataFrame:
    """Extract all sections from one file.

    Sections are broken into subsections if they are longer than `max_section_length`.
    Sections correspond to `section` HTML tags that have a headerlink attached.
    """
    filepath = os.path.join(root_dir, file)
    with open(filepath, "r") as f:
        source = f.read()

    soup = BeautifulSoup(source, "html.parser")
    parser = parser_cls(soup, base_url, root_dir, filepath, min_section_length, max_section_length)

    sections = []
    urls = []
    names = []
    for section in parser.parse():
        sections.append(section.text)
        urls.append(section.url)
        names.append(section.name)

    documents_df = pd.DataFrame.from_dict({"title": names, "url": urls, "content": sections})

    return documents_df


def get_all_documents(
    root_dir: str,
    base_url: str,
    parser_cls: Type[Parser],
    min_section_length: int = 100,
    max_section_length: int = 2000,
) -> pd.DataFrame:
    """Parse all HTML files in `root_dir`, and extract all sections.

    Sections are broken into subsections if they are longer than `max_section_length`.
    Sections correspond to `section` HTML tags that have a headerlink attached.
    """
    files = glob.glob("**/*.html", root_dir=root_dir, recursive=True)

    dfs = []
    for file in tqdm(files):
        try:
            df = get_document(root_dir, file, base_url, parser_cls, min_section_length, max_section_length)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {file} due to the following error: {e}")
            continue

    documents_df = pd.concat(dfs, ignore_index=True)

    return documents_df
