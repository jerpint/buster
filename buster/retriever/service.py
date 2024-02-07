import logging
from typing import List, Optional

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
        pinecone_index: str,
        pinecone_namespace: str,
        mongo_uri: str,
        mongo_db_name: str,
        **kwargs,
    ):
        """
        Initializes a ServiceRetriever instance.

        The ServiceRetriever is a hybrid retrieval combining pinecone and mongodb services.

        Pinecone is exclusively used as a vector store.
        The id of the pinecone vectors are used as a key in the mongodb database to store its associated metadata.

        Args:
            pinecone_api_key: The API key for Pinecone.
            pinecone_env: The environment for Pinecone.
            pinecone_index: The name of the Pinecone index.
            pinecone_namespace: The namespace for Pinecone.
            mongo_uri: The URI for MongoDB.
            mongo_db_name: The name of the MongoDB database.
        """
        super().__init__(**kwargs)

        pc = pinecone.Pinecone(api_key=pinecone_api_key)

        self.index = pc.Index(pinecone_index)
        self.namespace = pinecone_namespace

        self.client = MongoClient(mongo_uri, server_api=ServerApi("1"))
        self.db = self.client[mongo_db_name]

    def get_source_id(self, source: str) -> str:
        """Get the id of a source. Returns an empty string if the source does not exist.

        Args:
            source: The name of the source.

        Returns:
            The id of the source.
        """
        source_pointer = self.db.sources.find_one({"name": source})
        return "" if source_pointer is None else str(source_pointer["_id"])

    def get_documents(self, source: Optional[str] = None) -> pd.DataFrame:
        """Get all current documents from a given source.

        Args:
            source: The name of the source. Defaults to None.

        Returns:
            A DataFrame containing all the documents. If the source does not exist, returns an empty DataFrame.
        """
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
        """Get the display name of a source.

        Args:
            source: The name of the source.

        Returns:
            The display name of the source.
        """
        if source is None:
            return ALL_SOURCES
        else:
            display_name = self.db.sources.find_one({"name": source})["display_name"]
            return display_name

    def get_topk_documents(self, query: str, sources: Optional[List[str]], top_k: int) -> pd.DataFrame:
        """Get the top k documents matching a query from the specified sources.

        Args:
            query: The query string.
            sources: The list of source names to search. Defaults to None.
            top_k: The number of top matches to return.

        Returns:
            A DataFrame containing the top k matching documents.
        """
        if sources is None:
            filter = None
        else:
            filter = {"source": {"$in": sources}}
            source_exists = self.db.sources.find_one({"name": {"$in": sources}})
            if source_exists is None:
                logger.warning(f"Sources {sources} do not exist. Returning empty dataframe.")
                return pd.DataFrame()

        query_embedding = self.embedding_fn(query)
        sparse_query_embedding = self.sparse_embedding_fn(query) if self.sparse_embedding_fn is not None else None

        if isinstance(query_embedding, np.ndarray):
            # pinecone expects a list of floats, so convert from ndarray if necessary
            query_embedding = query_embedding.tolist()

        # Pinecone retrieval
        matches = self.index.query(
            vector=query_embedding,
            sparse_vector=sparse_query_embedding,
            top_k=top_k,
            filter=filter,
            include_values=True,
            namespace=self.namespace,
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
