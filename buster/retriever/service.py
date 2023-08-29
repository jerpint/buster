import logging

import numpy as np
import pandas as pd
import pinecone
from bson.objectid import ObjectId
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from buster.retriever.base import ALL_SOURCES, Retriever

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ServiceRetriever(Retriever):
    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_env: str,
        pinecone_index: str,
        pinecone_namespace: str,
        mongo_uri: str,
        mongo_db_name: str,
        **kwargs,
    ):
        super().__init__(**kwargs)

        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

        self.index = pinecone.Index(pinecone_index)
        self.namespace = pinecone_namespace

        self.client = MongoClient(mongo_uri, server_api=ServerApi("1"))
        self.db = self.client[mongo_db_name]

    def get_source_id(self, source: str) -> str:
        """Get the id of a source. Returns empty string if the source does not exist."""
        source_pointer = self.db.sources.find_one({"name": source})
        return "" if source_pointer is None else str(source_pointer["_id"])

    def get_documents(self, source: str = None) -> pd.DataFrame:
        """Get all current documents from a given source.

        If source is None, returns all documents. If source does not exist, returns empty dataframe."""

        if source is None:
            # No source specified, return all documents
            documents = self.db.documents.find()
        else:
            assert isinstance(source, str), "source must be a valid string."
            source_id = self.get_source_id(source)

            if source_id == "":
                logger.warning(f"{source=} not found.")

            documents = self.db.documents.find({"source_id": source_id})

        return pd.DataFrame(list(documents))

    def get_source_display_name(self, source: str) -> str:
        """Get the display name of a source."""
        if source is None:
            return ALL_SOURCES
        else:
            display_name = self.db.sources.find_one({"name": source})["display_name"]
            return display_name

    def get_topk_documents(self, query: str, source: str, top_k: int) -> pd.DataFrame:
        if source is None:
            filter = None
        else:
            filter = {"source": {"$eq": source}}
            source_exists = self.db.sources.find_one({"name": source})
            if source_exists is None:
                logger.warning(f"Source {source} does not exist. Returning empty dataframe.")
                return pd.DataFrame()

        query_embedding = self.get_embedding(query, model=self.embedding_model)

        if isinstance(query_embedding, np.ndarray):
            # pinecone expects a list of floats, so convert from ndarray if necessary
            query_embedding = query_embedding.tolist()

        # Pinecone retrieval
        matches = self.index.query(
            query_embedding, top_k=top_k, filter=filter, include_values=True, namespace=self.namespace
        )["matches"]
        matching_ids = [ObjectId(match.id) for match in matches]
        matching_scores = {match.id: match.score for match in matches}
        matching_embeddings = {match.id: match.values for match in matches}

        if len(matching_ids) == 0:
            return pd.DataFrame()

        # MongoDB retrieval
        matched_documents = self.db.documents.find({"_id": {"$in": matching_ids}})
        matched_documents = pd.DataFrame(list(matched_documents))

        # add additional information from matching
        matched_documents["similarity"] = matched_documents["_id"].apply(lambda x: matching_scores[str(x)])
        matched_documents["embedding"] = matched_documents["_id"].apply(lambda x: matching_embeddings[str(x)])

        # sort by similarity
        matched_documents = matched_documents.sort_values(by="similarity", ascending=False, ignore_index=True)

        return matched_documents
