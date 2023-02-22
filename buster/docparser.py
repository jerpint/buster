import glob
import os

import numpy as np
import pandas as pd
import tiktoken
from bs4 import BeautifulSoup
from openai.embeddings_utils import get_embedding

from buster.db import DocumentsDB
from buster.parser import HuggingfaceParser, Parser, SphinxParser

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_ENCODING = "cl100k_base"  # this the encoding for text-embedding-ada-002


PICKLE_EXTENSIONS = [".gz", ".bz2", ".zip", ".xz", ".zst", ".tar", ".tar.gz", ".tar.xz", ".tar.bz2"]


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
}


def get_all_documents(
    root_dir: str, base_url: str, parser: Parser, min_section_length: int = 100, max_section_length: int = 2000
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
        soup_parser = parser(soup, base_url, file, min_section_length, max_section_length)
        sections_file, urls_file, names_file = soup_parser.parse()

        sections.extend(sections_file)
        urls.extend(urls_file)
        names.extend(names_file)

    documents_df = pd.DataFrame.from_dict({"title": names, "url": urls, "content": sections})

    return documents_df


def get_file_extension(filepath: str) -> str:
    return os.path.splitext(filepath)[1]


def write_documents(filepath: str, documents_df: pd.DataFrame, source: str = ""):
    ext = get_file_extension(filepath)

    if ext == ".csv":
        documents_df.to_csv(filepath, index=False)
    elif ext in PICKLE_EXTENSIONS:
        documents_df.to_pickle(filepath)
    elif ext == ".db":
        db = DocumentsDB(filepath)
        db.write_documents(source, documents_df)
    else:
        raise ValueError(f"Unsupported format: {ext}.")


def read_documents(filepath: str, source: str = "") -> pd.DataFrame:
    ext = get_file_extension(filepath)

    if ext == ".csv":
        df = pd.read_csv(filepath)

        if "embedding" in df.columns:
            df["embedding"] = df.embedding.apply(eval).apply(np.array)
    elif ext in PICKLE_EXTENSIONS:
        df = pd.read_pickle(filepath)

        if "embedding" in df.columns:
            df["embedding"] = df.embedding.apply(np.array)
    elif ext == ".db":
        db = DocumentsDB(filepath)
        df = db.get_documents(source)
    else:
        raise ValueError(f"Unsupported format: {ext}.")

    return df


def compute_n_tokens(df: pd.DataFrame) -> pd.DataFrame:
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    # TODO are there unexpected consequences of allowing endoftext?
    df["n_tokens"] = df.content.apply(lambda x: len(encoding.encode(x, allowed_special={"<|endoftext|>"})))
    return df


def precompute_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    df["embedding"] = df.content.apply(lambda x: np.asarray(get_embedding(x, engine=EMBEDDING_MODEL), dtype=np.float32))
    return df


def generate_embeddings(filepath: str, output_file: str, source: str) -> pd.DataFrame:
    # Get all documents and precompute their embeddings
    df = read_documents(filepath, source)
    df = compute_n_tokens(df)
    df = precompute_embeddings(df)
    write_documents(output_file, source, df)
    return df
