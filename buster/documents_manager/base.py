import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import openai
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

EMBEDDING_MODEL = "text-embedding-ada-002"
REQUIRED_COLUMNS = ["url", "title", "content", "source"]


def get_embedding_openai(text: str):
    model = EMBEDDING_MODEL
    text = text.replace("\n", " ")
    try:
        return np.array(openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"], dtype=np.float32)
    except:
        # This rarely happens with the API but in the off chance it does, will allow us not to loose the progress.
        return None


@dataclass
class DocumentsManager(ABC):
    def _check_required_columns(self, df: pd.DataFrame):
        """Each entry in the df is expected to have the columns in self.required_columns"""
        if not all(col in df.columns for col in self.required_columns):
            raise ValueError(f"DataFrame is missing one or more of {self.required_columns=}")

    def _compute_embeddings(self, df: pd.DataFrame) -> pd.Series:
        """Compute the embeddings of a series, each entry expected to be a string.

        Override this method with an other embedding function if necessary.
        Returns a Series with the actual embeddings."""
        embeddings = df.content.progress_apply(get_embedding_openai)
        df["embedding"] = embeddings

        # saves a temporary file with the pre-computed embeddings.
        # This is helpful to have in the future to avoid having to recompute embeddings.
        filename = "chunks_with_embeddings.csv"
        df.to_csv(filename)
        logger.info(f"Finished computing embeddings, saved csv with embeddings to: {filename}")
        return embeddings

    def add(self, df: pd.DataFrame):
        """Write all documents from the DataFrame into the db as a new version."""

        self._check_required_columns(df)

        # Check if embeddings are present, computes them if not
        if "embedding" not in df.columns:
            logger.info(f"Computing embeddings for {len(df)} documents...")
            df["embedding"] = self._compute_embeddings(df)

        else:
            logger.info("Embeddings already present, skipping computation of embeddings")

        self._add_documents(df)

    @abstractmethod
    def _add_documents(self, df: pd.DataFrame):
        """Abstract method to be implemented by each inherited member.

        This method should handle the actual process of adding documents to the database.
        """
        ...
