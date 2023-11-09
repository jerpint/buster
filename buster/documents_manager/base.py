import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from buster.llm_utils import compute_embeddings_parallelized, get_openai_embedding

client = OpenAI()

tqdm.pandas()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def split_df_by_nans(df: pd.DataFrame, column: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits a DataFrame into two DataFrames, one with rows containing NaN and the other without NaNs.

    Args:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    (pd.DataFrame, pd.DataFrame): A tuple of two DataFrames, the first with NaN rows, the second without NaNs.
    """
    df_with_nans = df[df[column].isna()]
    df_without_nans = df.dropna()
    return df_without_nans, df_with_nans


@dataclass
class DocumentsManager(ABC):
    def __init__(self, required_columns: Optional[list[str]] = None):
        """
        Constructor for DocumentsManager class.

        Parameters:
            required_columns (Optional[list[str]]): A list of column names that are required for the dataframe to contain.
                                                    If None, no columns are enforced.
        """

        self.required_columns = required_columns

    def _check_required_columns(self, df: pd.DataFrame):
        """Each entry in the df is expected to have the columns in self.required_columns"""
        if not all(col in df.columns for col in self.required_columns):
            raise ValueError(f"DataFrame is missing one or more of {self.required_columns=}")

    def _checkpoint_csv(self, df, csv_filename: str, csv_overwrite: bool = True):
        import os

        if csv_overwrite:
            df.to_csv(csv_filename)
            logger.debug(f"Saved DataFrame to {csv_filename}")

        else:
            if os.path.exists(csv_filename):
                # append to existing file
                append_df = pd.read_csv(csv_filename)
                append_df = pd.concat([append_df, df])
            else:
                # will create the new file
                append_df = df.copy()
            append_df.to_csv(csv_filename)
            logger.debug(f"Appending DataFrame to {csv_filename}")

    def add(
        self,
        df: pd.DataFrame,
        num_workers: int = 16,
        embedding_fn: callable = get_openai_embedding,
        csv_embeddings_filename: Optional[str] = None,
        csv_errors_filename: Optional[str] = None,
        csv_overwrite: bool = True,
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

            csv_embeddings_filename: (str, optional) = Path to save a copy of the dataframe with computed embeddings for later use.
            csv_overwrite: (bool, optional) = If csv_filename is specified, whether to overwrite the file with a new file.
            **add_kwargs: Additional keyword arguments to be passed to the '_add_documents' method.


        """

        if self.required_columns is not None:
            self._check_required_columns(df)

        # Check if embeddings are present, computes them if not
        if "embedding" not in df.columns:
            df["embedding"] = compute_embeddings_parallelized(df, embedding_fn=embedding_fn, num_workers=num_workers)

        # errors with embeddings computation will be NaNs, so we filter them out and the user can recompute them later on...
        df, df_errors = split_df_by_nans(df, column="embedding")

        if len(df_errors) > 0:
            logger.warning(f"{len(df_errors)} errors have occured during embedding generation.")
            if csv_errors_filename is not None:
                self._checkpoint_csv(df_errors, csv_filename=csv_errors_filename, csv_overwrite=csv_overwrite)

        if csv_embeddings_filename is not None:
            self._checkpoint_csv(df, csv_filename=csv_embeddings_filename, csv_overwrite=csv_overwrite)

        self._add_documents(df, **add_kwargs)

    def batch_add(
        self,
        df: pd.DataFrame,
        batch_size: int = 3000,
        min_time_interval: int = 60,
        num_workers: int = 16,
        embedding_fn: callable = get_openai_embedding,
        csv_filename: Optional[str] = None,
        csv_overwrite: bool = False,
        **add_kwargs,
    ):
        """
        This function takes a DataFrame and adds its data to a DataManager instance in batches.
        It ensures that a minimum time interval is maintained between successive batches
        to prevent timeouts or excessive load. This is useful for APIs like openAI with rate limits.

        Args:
            df (pandas.DataFrame): The input DataFrame containing data to be added.
            batch_size (int, optional): The size of each batch. Defaults to 3000.
            min_time_interval (int, optional): The minimum time interval (in seconds) between batches.
                                            Defaults to 60.
            num_workers (int, optional): The number of parallel workers to use when adding data.
                                        Defaults to 32.
            embedding_fn (callable, optional): A function that computes embeddings for a given input string.
                Default is 'get_embedding_openai' which uses the text-embedding-ada-002 model.
            csv_filename: (str, optional) = Path to save a copy of the dataframe with computed embeddings for later use.
            csv_overwrite: (bool, optional) = If csv_filename is specified, whether to overwrite the file with a new file.
                When using batches, set to False to keep all embeddings in the same file. You may want to manually remove the file if experimenting.

            **add_kwargs: Additional keyword arguments to be passed to the '_add_documents' method.

        Returns:
            None
        """
        total_batches = (len(df) // batch_size) + 1

        logger.info(f"Adding {len(df)} documents with {batch_size=} for {total_batches=}")

        for batch_idx in range(total_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
            start_time = time.time()

            # Calculate batch indices and extract batch DataFrame
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            # Add the batch data to using specified parameters
            self.add(
                batch_df,
                num_workers=num_workers,
                csv_filename=csv_filename,
                csv_overwrite=csv_overwrite,
                embedding_fn=embedding_fn,
                **add_kwargs,
            )

            elapsed_time = time.time() - start_time

            # Sleep to ensure the minimum time interval is maintained
            # Only sleep if it's not the last iteration
            if batch_idx < total_batches - 1:
                sleep_time = max(0, min_time_interval - elapsed_time)
                if sleep_time > 0:
                    logger.info(f"Sleeping for {round(sleep_time)} seconds...")
                    time.sleep(sleep_time)

        logger.info("All batches processed.")

    @abstractmethod
    def _add_documents(self, df: pd.DataFrame, **add_kwargs):
        """Abstract method to be implemented by each inherited member.

        This method should handle the actual process of adding documents to the database.
        """
        ...
