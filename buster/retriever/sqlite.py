import sqlite3
from pathlib import Path

import pandas as pd

import buster.documents.sqlite.schema as schema
from buster.retriever.base import Retriever


class SQLiteRetriever(Retriever):
    """Simple SQLite database for retrieval of documents.

    The database is just a file on disk. It can store documents from different sources, and it 
    can store multiple versions of the same document (e.g. if the document is updated).

    Example:
        >>> db = DocumentsDB("/path/to/the/db.db")
        >>> df = db.get_documents("source")
    """
    def __init__(self, db_path: sqlite3.Connection | str):
        if isinstance(db_path, (str, Path)):
            self.db_path = db_path
            self.conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
        else:
            self.db_path = None
            self.conn = db_path
        schema.initialize_db(self.conn)
        schema.setup_db(self.conn)

    def __del__(self):
        if self.db_path is not None:
            self.conn.close()

    def get_documents(self, source: str) -> pd.DataFrame:
        """Get all current documents from a given source."""
        # Execute the SQL statement and fetch the results.
        if source is '':
            results = self.conn.execute("SELECT * FROM documents")
        else:
            results = self.conn.execute("SELECT * FROM documents WHERE source = ?", (source,))
        rows = results.fetchall()

        # Convert the results to a pandas DataFrame
        df = pd.DataFrame(rows, columns=[description[0] for description in results.description])
        return df