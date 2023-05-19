import logging

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
        mongo_uri: str,
        mongo_db_name: str,
        **kwargs,
    ):
        super().__init__(**kwargs)

        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

        self.index = pinecone.Index(pinecone_index)

        self.client = MongoClient(mongo_uri, server_api=ServerApi("1"))
        self.db = self.client[mongo_db_name]

    def get_source_id(self, source: str) -> str:
        """Get the id of a source."""
        return str(self.db.sources.find_one({"name": source})["_id"])

    def get_documents(self, source: str) -> pd.DataFrame:
        """Get all current documents from a given source."""
        source_id = self.get_source_id(source)
        return self.db.documents.find({"source_id": source_id})

    def get_source_display_name(self, source: str) -> str:
        """Get the display name of a source."""
        if source == "":
            return ALL_SOURCES
        else:
            display_name = self.db.sources.find_one({"name": source})["display_name"]
            return display_name

    def retrieve(self, query_embedding: list[float], top_k: int, source: str = None) -> pd.DataFrame:
        if source is "" or source is None:
            filter = None
        else:
            filter = {"source": {"$eq": source}}
            source_exists = self.db.sources.find_one({"name": source})
            if source_exists is None:
                logger.warning(f"Source {source} does not exist. Returning empty dataframe.")
                return pd.DataFrame()

        # Pinecone retrieval
        matches = self.index.query(query_embedding, top_k=top_k, filter=filter)["matches"]
        matching_ids = [ObjectId(match.id) for match in matches]
        matching_scores = {match.id: match.score for match in matches}

        if len(matching_ids) == 0:
            return pd.DataFrame()

        # MongoDB retrieval
        matched_documents = self.db.documents.find({"_id": {"$in": matching_ids}})
        matched_documents = pd.DataFrame(list(matched_documents))
        matched_documents["similarity"] = matched_documents["_id"].apply(lambda x: matching_scores[str(x)])

        return matched_documents
