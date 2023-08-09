import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from buster.documents_manager.base import get_embedding_openai

ALL_SOURCES = "All"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Retriever(ABC):
    def __init__(self, top_k, thresh, max_tokens, embedding_model, *args, **kwargs):
        self.top_k = top_k
        self.thresh = thresh
        self.max_tokens = max_tokens
        self.embedding_model = embedding_model

        # Add your access to documents in your own init

    @abstractmethod
    def get_documents(self, source: str = None) -> pd.DataFrame:
        """Get all current documents from a given source."""
        ...

    @abstractmethod
    def get_source_display_name(self, source: str) -> str:
        """Get the display name of a source.

        If source is None, returns all documents. If source does not exist, returns empty dataframe."""
        ...

    @staticmethod
    @lru_cache
    def get_embedding(query: str, model: str) -> np.ndarray:
        logger.info("generating embedding")
        return get_embedding_openai(query, model=model)

    @abstractmethod
    def get_topk_documents(self, query: str, source: str = None, top_k: int = None) -> pd.DataFrame:
        """Get the topk documents matching a user's query.

        If no matches are found, returns an empty dataframe."""
        ...

    def threshold_documents(self, matched_documents, thresh: float) -> pd.DataFrame:
        # filter out matched_documents using a threshold
        return matched_documents[matched_documents.similarity > thresh]

    def retrieve(self, query: str, source: str = None, top_k: int = None, thresh: float = None) -> pd.DataFrame:
        if top_k is None:
            top_k = self.top_k
        if thresh is None:
            thresh = self.thresh

        matched_documents = self.get_topk_documents(query=query, source=source, top_k=top_k)

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
