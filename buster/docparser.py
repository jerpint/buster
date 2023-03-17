import glob
import logging
import os
from typing import Type

import click
import numpy as np
import pandas as pd
import tiktoken
from bs4 import BeautifulSoup
from openai.embeddings_utils import get_embedding

from buster.documents import get_documents_manager_from_extension
from buster.parser import HuggingfaceParser, Parser, SphinxParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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


def compute_n_tokens(
    df: pd.DataFrame, embedding_encoding: str = EMBEDDING_ENCODING, col: str = "content"
) -> pd.DataFrame:
    """Counts the tokens in the content column and adds the count to a n_tokens column."""
    logger.info("Computing tokens counts...")
    encoding = tiktoken.get_encoding(encoding_name=embedding_encoding)
    # TODO are there unexpected consequences of allowing endoftext?
    df[f"n_tokens"] = df[col].apply(lambda x: len(encoding.encode(x, allowed_special={"<|endoftext|>"})))
    return df


def max_word_count(df: pd.DataFrame, max_words: int, col: str = "content") -> pd.DataFrame:
    """Trim the word count of an entry to max_words"""
    assert df[col].apply(lambda s: isinstance(s, str)).all(), "Column {col} must contain only strings"
    word_counts_before = df[col].apply(lambda x: len(x.split(" ")))
    df[col] = df[col].apply(lambda x: " ".join(x.split()[:max_words]))
    word_counts_after = df[col].apply(lambda x: len(x.split(" ")))

    trimmed = df[word_counts_before == word_counts_after]
    logger.info(f"trimmed {len(trimmed)} documents to {max_words} words.")

    return df


def compute_embeddings(df: pd.DataFrame, engine: str = EMBEDDING_MODEL) -> pd.DataFrame:
    logger.info(f"Computing embeddings for {len(df)} documents...")
    df["embedding"] = df.content.apply(lambda x: np.asarray(get_embedding(x, engine=engine), dtype=np.float32))
    logger.info(f"Done computing embeddings for {len(df)} documents.")
    return df


def generate_embeddings_parser(root_dir: str, output_filepath: str, source: str) -> pd.DataFrame:
    documents = get_all_documents(root_dir, supported_docs[source]["base_url"], supported_docs[source]["parser"])
    return generate_embeddings(documents, output_filepath)


def documents_to_db(documents: pd.DataFrame, output_filepath: str):
    logger.info("Preparing database...")
    sources = documents["source"].unique()
    for source in sources:
        documents_manager = get_documents_manager_from_extension(output_filepath)(output_filepath)
        documents_manager.add(source, documents)
    logger.info(f"Documents saved to: {output_filepath}")


def generate_embeddings(
    documents: pd.DataFrame,
    output_filepath: str = "documents.db",
    max_words=500,
    embedding_engine: str = EMBEDDING_MODEL,
) -> pd.DataFrame:
    # check that we have the appropriate columns in our dataframe

    assert set(required_cols := ["content", "title", "url"]).issubset(
        set(documents.columns)
    ), f"Your datafraem must contain {required_cols}."

    # Get all documents and precompute their embeddings
    documents = max_word_count(documents, max_words=max_words)
    documents = compute_n_tokens(documents)
    documents = compute_embeddings(documents, engine=embedding_engine)

    # save the documents to a db for later use
    documents_to_db(documents, output_filepath)

    return documents


@click.command()
@click.argument("documents-csv")
@click.option(
    "--output-filepath", default="documents.db", help='Where your database will be saved. Default is "documents.db"'
)
@click.option(
    "--max-words", default=500, help="Number of maximum allowed words per document, excess is trimmed. Default is 500"
)
@click.option(
    "--embeddings-engine", default=EMBEDDING_MODEL, help=f"Embedding model to use. Default is {EMBEDDING_MODEL}"
)
def main(documents_csv: str, output_filepath: str, max_words: int, embeddings_engine: str):
    documents = pd.read_csv(documents_csv)
    documents = generate_embeddings(documents, output_filepath, max_words, embeddings_engine)


if __name__ == "__main__":
    main()
