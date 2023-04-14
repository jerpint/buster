import os

import pandas as pd
import pinecone
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from buster.documents.base import DocumentsManager


class DocumentsService(DocumentsManager):
    """Manager to use in production. Mixed Pinecone and MongoDB backend."""

    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_env: str,
        pinecone_index: str,
        mongo_uri: str,
        mongo_db_name: str,
        **kwargs
    ):
        super().__init__(**kwargs)

        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

        self.index = pinecone.Index(pinecone_index)

        self.client = MongoClient(mongo_uri, server_api=ServerApi("1"))
        self.db = self.client[mongo_db_name]

    def add(self, source: str, df: pd.DataFrame):
        """Write all documents from the dataframe into the db as a new version."""
        source_exists = self.db.sources.find_one({"name": source})
        if source_exists is None:
            self.db.sources.insert_one({"name": source})

        source_id = str(self.db.sources.find_one({"name": source})["_id"])

        for _, row in df.iterrows():
            document = {
                "title": row["title"],
                "url": row["url"],
                "content": row["content"],
                "n_tokens": row["n_tokens"],
                "source_id": source_id,
            }
            document_id = str(self.db.documents.insert_one(document).inserted_id)
            self.index.upsert([(document_id, row["embedding"].tolist(), {"source": source})])

    def update_source(self, source: str, display_name: str = None, note: str = None):
        """Update the display name and/or note of a source. Also create the source if it does not exist."""
        self.db.sources.update_one(
            {"name": source}, {"$set": {"display_name": display_name, "note": note}}, upsert=True
        )
