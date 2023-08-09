import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import openai
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

tqdm.pandas()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REQUIRED_COLUMNS = ["url", "title", "content", "source"]


def get_embedding_openai(text: str, model="text-embedding-ada-002") -> np.ndarray:
    text = text.replace("\n", " ")
    try:
        return np.array(openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"], dtype=np.float32)
    except:
        # This rarely happens with the API but in the off chance it does, will allow us not to loose the progress.
        logger.warning("Embedding failed to compute.")
        return None


def compute_embeddings_parallelized(df: pd.DataFrame, embedding_fn: callable, num_workers: int) -> pd.Series:
    """Compute the embeddings on the 'content' column of a DataFrame in parallel.

    This method calculates embeddings for the entries in the 'content' column of the provided DataFrame using the specified
    embedding function. The 'content' column is expected to contain strings or textual data. The method processes the
    embeddings in parallel using the number of workers specified.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to compute embeddings for.
        embedding_fn (callable): A function that computes embeddings for a given input string.
        num_workers (int): The number of parallel workers to use for computing embeddings.

    Returns:
        pd.Series: A Series containing the computed embeddings for each entry in the 'content' column.
    """

    logger.info(f"Computing embeddings of {len(df)} chunks. Using {num_workers=}")
    embeddings = process_map(embedding_fn, df.content.to_list(), max_workers=num_workers)

    logger.info(f"Finished computing embeddings")
    return embeddings


@dataclass
class DocumentsManager(ABC):
    def __init__(self, required_columns: list[str], csv_checkpoint: Optional[str] = None):
        self.required_columns = required_columns
        self.csv_checkpoint = csv_checkpoint

    def _check_required_columns(self, df: pd.DataFrame):
        """Each entry in the df is expected to have the columns in self.required_columns"""
        if not all(col in df.columns for col in self.required_columns):
            raise ValueError(f"DataFrame is missing one or more of {self.required_columns=}")

    def add(
        self,
        df: pd.DataFrame,
        num_workers: int = 16,
        embedding_fn: callable = get_embedding_openai,
        **add_kwargs,
    ):
        """Write documents from a DataFrame into the DocumentManager store.

        This method adds documents from the provided DataFrame to the database. It performs the following steps:
        1. Checks if the required columns are present in the DataFrame.
        2. Computes embeddings for the 'content' column if they are not already present.
        3. Optionally saves the DataFrame with computed embeddings to a CSV checkpoint.
        4. Calls the '_add_documents' method to add documents with embeddinsg to the DocumentsManager.

        Args:
            df (pd.DataFrame): The DataFrame containing the documents to be added.
            num_workers (int, optional): The number of parallel workers to use for computing embeddings. Default is 32.
            embedding_fn (callable, optional): A function that computes embeddings for a given input string.
                Default is 'get_embedding_openai' which uses the text-embedding-ada-002 model.
            **add_kwargs: Additional keyword arguments to be passed to the '_add_documents' method.


        """

        self._check_required_columns(df)

        # Check if embeddings are present, computes them if not
        if "embedding" not in df.columns:
            df["embedding"] = compute_embeddings_parallelized(df, embedding_fn=embedding_fn, num_workers=num_workers)

        if self.csv_checkpoint is not None:
            df.to_csv(self.csv_checkpoint)
            logger.info("Saving .csv with embeddings to {self.csv_checkpoint}")

        self._add_documents(df, **add_kwargs)

    @abstractmethod
    def _add_documents(self, df: pd.DataFrame, **add_kwargs):
        """Abstract method to be implemented by each inherited member.

        This method should handle the actual process of adding documents to the database.
        """
        ...
