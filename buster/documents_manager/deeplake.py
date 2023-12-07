import logging
from typing import Optional

import pandas as pd

from buster.utils import zip_contents

from .base import DocumentsManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DeepLakeDocumentsManager(DocumentsManager):
    def __init__(
        self,
        vector_store_path: str = "deeplake_store",
        required_columns: Optional[list[str]] = None,
        **vector_store_kwargs,
    ):
        """Initialize a DeepLakeDocumentsManager object.

        Args:
            vector_store_path: The path to the vector store.
            required_columns: A list of columns that are required in the dataframe.
            **vector_store_kwargs: Additional keyword arguments to pass to the VectorStore initializer.
        """
        from deeplake.core.vectorstore import VectorStore

        self.vector_store_path = vector_store_path
        self.required_columns = required_columns
        self.vector_store = VectorStore(
            path=self.vector_store_path,
            **vector_store_kwargs,
        )

    def __len__(self):
        """Get the number of documents in the vector store.

        Returns:
            The number of documents in the vector store.
        """
        return len(self.vector_store)

    @classmethod
    def _extract_metadata(cls, df: pd.DataFrame) -> dict:
        """Extract metadata from the dataframe in DeepLake dict format.

        Args:
            df: The dataframe from which to extract metadata.

        Returns:
            The extracted metadata in DeepLake dict format.
        """
        # Ignore the content and embedding column for metadata
        df = df.drop(columns=["content", "embedding"], errors="ignore")

        columns = list(df.columns)

        metadata = df.apply(
            lambda x: {col: x[col] for col in columns},
            axis=1,
        ).to_list()
        return metadata

    def _add_documents(self, df: pd.DataFrame, **add_kwargs):
        """Write all documents from the dataframe into the vector store as a new version.

        Each entry in the dataframe is expected to have at least the following columns:
        ["content", "embedding"]

        Embeddings will have been precomputed in the self.add() method, which calls this one.

        Args:
            df: The dataframe containing the documents to add.
            **add_kwargs: Additional keyword arguments to pass to the add method of the vector store.
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
        """Zip the contents of the vector store path folder to a .zip file in the output path.

        Args:
            output_path: The path where the zip file should be created.

        Returns:
            The path to the created zip file.
        """
        vector_store_path = self.vector_store_path
        logger.info(f"Compressing {vector_store_path}...")
        zip_file_path = zip_contents(input_path=vector_store_path, output_path=output_path)
        logger.info(f"Compressed {vector_store_path} to {zip_file_path}.")
        return zip_file_path
