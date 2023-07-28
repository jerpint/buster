import logging
from typing import Optional

import openai
import pandas as pd

from buster.utils import zip_contents

from .base import DocumentsManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def openai_embedding_function(texts, model="text-embedding-ada-002"):
    if isinstance(texts, str):
        texts = [texts]

    texts = [t.replace("\n", " ") for t in texts]
    return [data["embedding"] for data in openai.Embedding.create(input=texts, model=model)["data"]]


def extract_metadata(df: pd.DataFrame) -> dict:
    """extract the metadata from the dataframe in deeplake dict format"""

    columns = list(df.columns)

    metadata = df.apply(
        lambda x: {col: x[col] for col in columns},
        axis=1,
    ).to_list()
    return metadata


def read_csv(filename: str):
    """Assumes a pre-chunked csv file is provided with expected columns."""
    df = pd.read_csv(filename)
    for col in ["url", "source", "title", "content"]:
        assert col in df.columns
    return df


class DeepLakeDocumentsManager(DocumentsManager):
    def __init__(self, vector_store_path, **vector_store_kwargs):
        from deeplake.core.vectorstore import VectorStore

        self.vector_store_path = vector_store_path
        self.vector_store = VectorStore(
            path=self.vector_store_path,
            **vector_store_kwargs,
        )

    def update_source(self, source: str, display_name: str = None, note: str = None):
        """Update the display name and/or note of a source. Also create the source if it does not exist."""
        raise NotImplementedError()

    def add(self, df: pd.DataFrame):
        """Write all documents from the dataframe into the db as a new version.

        Each entry in the df is expected to have at least the following columns:
        ["url", "source", "title", "content"]

        You can pass precomputed embeddings in an "embedding" column, otherwise it will compute embeddings for you.
        """
        for col in ["url", "source", "title", "content"]:
            assert col in df.columns, "Check that all required columns are present."
        # extract the chunked text + metadata
        metadata = extract_metadata(
            df.drop(columns=["content", "embedding"]),  # ignoring the content and embedding column for metadata
        )
        chunked_text = df.content.to_list()

        if "embedding" in df.columns:
            logger.info("embeddings provided in csv, not computing them.")
            embeddings = df.embedding.to_list()
            self.vector_store.add(
                text=chunked_text,
                embedding=embeddings,
                metadata=metadata,
            )

        else:
            # compute and add the embeddings
            logger.info("Computing and adding embeddings...")
            self.vector_store.add(
                text=chunked_text,
                embedding_function=openai_embedding_function,
                embedding_data=chunked_text,
                metadata=metadata,
            )

    def to_zip(self, output_path: str = "."):
        """Zip the contents of the vector_store_path folder to a .zip file in output_path."""
        vector_store_path = self.vector_store_path
        logger.info(f"Compressing {vector_store_path}...")
        zip_file_path = zip_contents(input_path=vector_store_path, output_path=output_path)
        logger.info(f"Compressed {vector_store_path} to {zip_file_path}.")
        return zip_file_path
