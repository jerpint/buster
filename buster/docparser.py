import glob
import os
from typing import Type

import numpy as np
import pandas as pd
import tiktoken
from bs4 import BeautifulSoup
from openai.embeddings_utils import get_embedding

from buster.documents import get_documents_manager_from_extension
from buster.parser import HuggingfaceParser, Parser, SphinxParser

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_ENCODING = "cl100k_base"  # this the encoding for text-embedding-ada-002


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

    sections = []
    urls = []
    names = []
    for file in files:
        filepath = os.path.join(root_dir, file)
        with open(filepath, "r") as f:
            source = f.read()

        soup = BeautifulSoup(source, "html.parser")
        parser = parser_cls(soup, base_url, file, min_section_length, max_section_length)
        # sections_file, urls_file, names_file =
        for section in parser.parse():
            sections.append(section.text)
            urls.append(section.url)
            names.append(section.name)

    documents_df = pd.DataFrame.from_dict({"title": names, "url": urls, "content": sections})

    return documents_df


def compute_n_tokens(df: pd.DataFrame) -> pd.DataFrame:
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    # TODO are there unexpected consequences of allowing endoftext?
    df["n_tokens"] = df.content.apply(lambda x: len(encoding.encode(x, allowed_special={"<|endoftext|>"})))
    return df


def precompute_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    df["embedding"] = df.content.apply(lambda x: np.asarray(get_embedding(x, engine=EMBEDDING_MODEL), dtype=np.float32))
    return df


def generate_embeddings(root_dir: str, output_filepath: str, source: str) -> pd.DataFrame:
    # Get all documents and precompute their embeddings
    documents = get_all_documents(root_dir, supported_docs[source]["base_url"], supported_docs[source]["parser"])
    documents = compute_n_tokens(documents)
    documents = precompute_embeddings(documents)

    documents_manager = get_documents_manager_from_extension(output_filepath)(output_filepath)
    documents_manager.add(source, documents)

    return documents
