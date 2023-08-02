import logging
from typing import Optional

import openai
import pandas as pd

from buster.utils import zip_contents

from .base import DocumentsManager
from .base import get_embedding_openai, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def read_csv(filename: str):
    """Assumes a pre-chunked csv file is provided with expected columns."""
    df = pd.read_csv(filename)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"{col} not found in csv."
    return df


class DeepLakeDocumentsManager(DocumentsManager):
    def __init__(self, vector_store_path: str ="deeplake_store", required_columns=REQUIRED_COLUMNS, **vector_store_kwargs):
        from deeplake.core.vectorstore import VectorStore

        self.vector_store_path = vector_store_path
        self.required_columns = required_columns
        self.vector_store = VectorStore(
            path=self.vector_store_path,
            **vector_store_kwargs,
        )

    def _compute_embeddings(self, ser: pd.Series) -> pd.Series:
        embeddings = ser.apply(lambda x: get_embedding_openai(x, model="text-embedding-ada-002"))
        return embeddings

    @classmethod
    def _extract_metadata(cls, df: pd.DataFrame) -> dict:
        """extract the metadata from the dataframe in deeplake dict format"""

        # Ignore the content and embedding column for metadata
        df = df.drop(columns=["content", "embedding"], errors="ignore")

        columns = list(df.columns)

        metadata = df.apply(
            lambda x: {col: x[col] for col in columns},
            axis=1,
        ).to_list()
        return metadata

    def _add_documents(self, df: pd.DataFrame, **add_kwargs):
        """Write all documents from the dataframe into the db as a new version.

        Each entry in the df is expected to have at least the following columns:
        ["content", "embedding"]

        Embeddings will have been precomputed in the self.add() method, which calls this one.
        """
        # Embedding should already be computed in the .add method
        assert "embedding" in df.columns, "expected column=embedding in the dataframe"

        # extract the chunked text + metadata
        metadata = self._extract_metadata(df)

        chunked_text = df.content.to_list()

        embeddings = df.embedding.to_list()
        self.vector_store.add(
            text=chunked_text,
            embedding=embeddings,
            metadata=metadata,
            **add_kwargs,
        )

    def to_zip(self, output_path: str = "."):
        """Zip the contents of the vector_store_path folder to a .zip file in output_path."""
        vector_store_path = self.vector_store_path
        logger.info(f"Compressing {vector_store_path}...")
        zip_file_path = zip_contents(input_path=vector_store_path, output_path=output_path)
        logger.info(f"Compressed {vector_store_path} to {zip_file_path}.")
        return zip_file_path