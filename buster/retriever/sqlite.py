import os
import sqlite3
from pathlib import Path

import pandas as pd
from openai.embeddings_utils import cosine_similarity

import buster.documents.sqlite.schema as schema
from buster.retriever.base import ALL_SOURCES, Retriever


class SQLiteRetriever(Retriever):
    """Simple SQLite database for retrieval of documents.

    The database is just a file on disk. It can store documents from different sources, and it
    can store multiple versions of the same document (e.g. if the document is updated).

    Example:
        >>> db = DocumentsDB("/path/to/the/db.db")
        >>> df = db.get_documents("source")
    """

    def __init__(self, db_path: str | Path = None, connection: sqlite3.Connection = None, **kwargs):
        super().__init__(**kwargs)

        match sum([arg is not None for arg in [db_path, connection]]):
            # Check that only db_path or connection get specified
            case 0:
                raise ValueError("At least one of db_path or connection should be specified")
            case 2:
                raise ValueError("Only one of db_path and connection should be specified.")

        if connection is not None:
            self.conn = connection

        if db_path is not None:
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"{db_path=} specified, but file does not exist")
            self.db_path = db_path
            self.conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)

        schema.setup_db(self.conn)

    def __del__(self):
        if self.db_path is not None:
            self.conn.close()

    def get_documents(self, source: str = None) -> pd.DataFrame:
        """Get all current documents from a given source."""
        # Execute the SQL statement and fetch the results.
        if source is None:
            results = self.conn.execute("SELECT * FROM documents")
        else:
            results = self.conn.execute("SELECT * FROM documents WHERE source = ?", (source,))
        rows = results.fetchall()

        # Convert the results to a pandas DataFrame
        df = pd.DataFrame(rows, columns=[description[0] for description in results.description])
        return df

    def get_source_display_name(self, source: str) -> str:
        """Get the display name of a source."""
        if source is None:
            return ALL_SOURCES
        else:
            cur = self.conn.execute("SELECT display_name FROM sources WHERE name = ?", (source,))
            row = cur.fetchone()
            if row is None:
                raise KeyError(f'"{source}" is not a known source')
            (display_name,) = row
            return display_name

    def get_topk_documents(self, query: str, source: str = None, top_k: int = None) -> pd.DataFrame:
        query_embedding = self.get_embedding(query, engine=self.embedding_model)

        documents = self.get_documents(source)

        documents["similarity"] = documents.embedding.apply(lambda x: cosine_similarity(x, query_embedding))

        # sort the matched_documents by score
        matched_documents = documents.sort_values("similarity", ascending=False)

        # limit search to top_k matched_documents.
        top_k = len(matched_documents) if top_k == -1 else top_k
        matched_documents = matched_documents.head(top_k)

        return matched_documents
