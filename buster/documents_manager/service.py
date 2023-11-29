import logging

import pandas as pd
import pinecone
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from buster.documents_manager.base import DocumentsManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DocumentsService(DocumentsManager):
    """Manager to use in production. Mixed Pinecone and MongoDB backend."""

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

        self.mongo_db_name = mongo_db_name
        self.client = MongoClient(mongo_uri, server_api=ServerApi("1"))
        self.db = self.client[mongo_db_name]

    def __repr__(self):
        return "DocumentsService"

    def get_source_id(self, source: str) -> str:
        """Get the id of a source."""
        return str(self.db.sources.find_one({"name": source})["_id"])

    def _add_documents(self, df: pd.DataFrame):
        """Write all documents from the dataframe into the db as a new version."""

        use_sparse_vector = "sparse_embedding" in df.columns
        if use_sparse_vector:
            logger.info("Uploading sparse embeddings too.")

        for source in df.source.unique():
            source_exists = self.db.sources.find_one({"name": source})
            if source_exists is None:
                self.db.sources.insert_one({"name": source})

            source_id = self.get_source_id(source)

            df_source = df[df.source == source]
            to_upsert = []
            for row in df_source.to_dict(orient="records"):
                embedding = row["embedding"].tolist()
                if use_sparse_vector:
                    sparse_embedding = row["sparse_embedding"]

                document = row.copy()
                document.pop("embedding")
                if use_sparse_vector:
                    document.pop("sparse_embedding")
                document["source_id"] = source_id

                document_id = str(self.db.documents.insert_one(document).inserted_id)
                vector = {"id": document_id, "values": embedding, "metadata": {"source": source}}
                if use_sparse_vector:
                    vector["sparse_values"] = sparse_embedding

                to_upsert.append(vector)

            # Current (November 2023) Pinecone upload rules:
            # - Max 1000 vectors per batch
            # - Max 2 MB per batch
            # Sparse vectors are heavier, so we reduce the batch size when using them.
            MAX_PINECONE_BATCH_SIZE = 100 if use_sparse_vector else 1000
            for i in range(0, len(to_upsert), MAX_PINECONE_BATCH_SIZE):
                self.index.upsert(vectors=to_upsert[i : i + MAX_PINECONE_BATCH_SIZE], namespace=self.namespace)

    def update_source(self, source: str, display_name: str = None, note: str = None):
        """Update the display name and/or note of a source. Also create the source if it does not exist."""
        self.db.sources.update_one(
            {"name": source}, {"$set": {"display_name": display_name, "note": note}}, upsert=True
        )

    def delete_source(self, source: str) -> tuple[int, int]:
        """Delete a source and all its documents. Return if the source was deleted and the number of deleted documents."""
        source_id = self.get_source_id(source)

        # MongoDB
        source_deleted = self.db.sources.delete_one({"name": source}).deleted_count
        documents_deleted = self.db.documents.delete_many({"source_id": source_id}).deleted_count

        # Pinecone
        self.index.delete(filter={"source": source}, namespace=self.namespace)

        return source_deleted, documents_deleted

    def drop_db(self):
        """Drop the currently accessible database.

        For Pinecone, this means deleting everything in the namespace.
        For Mongo DB, this means dropping the database. However this needs to be done manually through the GUI.
        """
        confirmation = input("Dropping the database is irreversible. Are you sure you want to proceed? (y/N): ")

        if confirmation.strip().lower() == "y":
            self.index.delete(namespace=self.namespace, delete_all=True)

            logging.info(f"Deleted all documents from Pinecone namespace: {self.namespace=}")
            logging.info(f"The MongoDB database needs to be dropped manually: {self.mongo_db_name=}")
        else:
            logging.info("Operation cancelled.")
