import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
import pandas as pd

from buster.completers import UserInputs
from buster.llm_utils import get_openai_embedding

ALL_SOURCES = "All"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Retriever(ABC):
    def __init__(self, top_k: int, thresh: float, embedding_fn: Callable[[str], np.array] = None, *args, **kwargs):
        """Initializes a Retriever instance.

        Args:
          top_k: The maximum number of documents to retrieve.
          thresh: The similarity threshold for document retrieval.
          embedding_fn: The function to compute document embeddings.
          *args, **kwargs: Additional arguments and keyword arguments.
        """
        if embedding_fn is None:
            embedding_fn = get_openai_embedding

        self.top_k = top_k
        self.thresh = thresh
        self.embedding_fn = embedding_fn

        # Add your access to documents in your own init

    @abstractmethod
    def get_documents(self, source: Optional[str] = None) -> pd.DataFrame:
        """Get all current documents from a given source.

        Args:
          source: The source from which to retrieve documents. If None, retrieves documents from all sources.

        Returns:
          A pandas DataFrame containing the documents.
        """
        ...

    @abstractmethod
    def get_source_display_name(self, source: str) -> str:
        """Get the display name of a source.

        Args:
          source: The source for which to retrieve the display name.

        Returns:
          The display name of the source.

        If source is None, returns all documents. If source does not exist, returns empty dataframe.
        """
        ...

    def get_embedding(self, query: str) -> np.ndarray:
        """Generates the embedding of a query.

        Args:
          query: The query for which to generate the embedding.

        Returns:
          The embedding of the query as a NumPy array.
        """
        logger.info("generating embedding")
        return self.embedding_fn(query)

    @abstractmethod
    def get_topk_documents(self, query: str, source: Optional[str] = None, top_k: Optional[int] = None) -> pd.DataFrame:
        """Get the topk documents matching a user's query.

        Args:
          query: The user's query.
          source: The source from which to retrieve documents. If None, retrieves documents from all sources.
          top_k: The maximum number of documents to retrieve.

        Returns:
          A pandas DataFrame containing the topk matched documents.

        If no matches are found, returns an empty dataframe.
        """
        ...

    def threshold_documents(self, matched_documents: pd.DataFrame, thresh: float) -> pd.DataFrame:
        """Filters out matched documents using a similarity threshold.

        Args:
          matched_documents: The DataFrame containing the matched documents.
          thresh: The similarity threshold.

        Returns:
          A pandas DataFrame containing the filtered matched documents.
        """
        # filter out matched_documents using a threshold
        return matched_documents[matched_documents.similarity > thresh]

    def retrieve(
        self,
        user_inputs: UserInputs,
        sources: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        thresh: Optional[float] = None,
    ) -> pd.DataFrame:
        """Retrieves documents based on user inputs.

        Args:
          user_inputs: The user's inputs.
          sources: The sources from which to retrieve documents. If None, retrieves documents from all sources.
          top_k: The maximum number of documents to retrieve.
          thresh: The similarity threshold for document retrieval.

        Returns:
          A pandas DataFrame containing the retrieved documents.
        """
        if top_k is None:
            top_k = self.top_k
        if thresh is None:
            thresh = self.thresh

        query = user_inputs.current_input

        matched_documents = self.get_topk_documents(query=query, sources=sources, top_k=top_k)

        # log matched_documents to the console
        logger.info(f"matched documents before thresh: {matched_documents}")

        # No matches were found, simply return at this point
        if len(matched_documents) == 0:
            return matched_documents

        # otherwise, make sure we have the minimum required fields
        assert "similarity" in matched_documents.columns
        assert "embedding" in matched_documents.columns
        assert "content" in matched_documents.columns
        assert "title" in matched_documents.columns

        # filter out matched_documents using a threshold
        matched_documents = self.threshold_documents(matched_documents, thresh)

        logger.info(f"matched documents after thresh: {matched_documents}")

        return matched_documents
