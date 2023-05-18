from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
import logging

import pandas as pd
from openai.embeddings_utils import cosine_similarity, get_embedding

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

    @abstractmethod
    def get_documents(self, source: str) -> pd.DataFrame:
        """Get all current documents from a given source."""
        ...

    @abstractmethod
    def get_source_display_name(self, source: str) -> str:
        """Get the display name of a source."""
        ...

    @staticmethod
    @lru_cache
    def get_embedding(query: str, engine: str):
        logger.info("generating embedding")
        return get_embedding(query, engine=engine)

    def retrieve(self, query: str, source: str = None) -> pd.DataFrame:
        top_k = self.top_k
        thresh = self.thresh
        query_embedding = self.get_embedding(query, engine=self.embedding_model)

        documents = self.get_documents(source)

        documents["similarity"] = documents.embedding.apply(lambda x: cosine_similarity(x, query_embedding))

        # sort the matched_documents by score
        matched_documents = documents.sort_values("similarity", ascending=False)

        # limit search to top_k matched_documents.
        top_k = len(matched_documents) if top_k == -1 else top_k
        matched_documents = matched_documents.head(top_k)

        # log matched_documents to the console
        logger.info(f"matched documents before thresh: {matched_documents}")

        # filter out matched_documents using a threshold
        matched_documents = matched_documents[matched_documents.similarity > thresh]
        logger.info(f"matched documents after thresh: {matched_documents}")

        return matched_documents
