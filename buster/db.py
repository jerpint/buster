import sqlite3
import zlib

import numpy as np
import pandas as pd


documents_table = """CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    content TEXT NOT NULL,
    n_tokens INTEGER,
    embedding BLOB,
    current INTEGER
)"""

qa_table = """CREATE TABLE IF NOT EXISTS qa (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    prompt TEXT NOT NULL,
    answer TEXT NOT NULL,
    document_id_1 INTEGER,
    document_id_2 INTEGER,
    document_id_3 INTEGER,
    label_question INTEGER,
    label_answer INTEGER,
    testset INTEGER,
    FOREIGN KEY (document_id_1) REFERENCES documents (id),
    FOREIGN KEY (document_id_2) REFERENCES documents (id),
    FOREIGN KEY (document_id_3) REFERENCES documents (id)
)"""


class DocumentsDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self.__initialize()

    def __del__(self):
        self.conn.close()

    def __initialize(self):
        """Initialize the database."""
        self.cursor.execute(documents_table)
        self.cursor.execute(qa_table)
        self.conn.commit()

    def reset_document_source(self, source: str, df: pd.DataFrame):
        """Reset the current flag for all documents from a given source."""
        df = df.copy()

        # Prepare the rows
        df["source"] = source
        df["current"] = 1
        columns = ["source", "title", "url", "content", "current"]
        if "embedding" in df.columns:
            columns.extend(
                [
                    "n_tokens",
                    "embedding",
                ]
            )

            # ZLIB compress the embeddings
            df["embedding"] = df["embedding"].apply(lambda x: sqlite3.Binary(zlib.compress(x.tobytes())))

        data = df[columns].values.tolist()

        # Set `current` to 0 for all previous documents from that source
        self.cursor.execute("UPDATE documents SET current = 0 WHERE source = ?", (source,))

        # Insert the new documents
        insert_statement = f"INSERT INTO documents ({', '.join(columns)}) VALUES ({', '.join(['?']*len(columns))})"
        self.cursor.executemany(insert_statement, data)

        self.conn.commit()

    def get_documents(self, source: str) -> pd.DataFrame:
        """Get all current documents from a given source."""
        # Execute the SQL statement and fetch the results
        results = self.cursor.execute("SELECT * FROM documents WHERE source = ? AND current = 1", (source,))
        rows = results.fetchall()

        # Convert the results to a pandas DataFrame
        df = pd.DataFrame(rows, columns=[description[0] for description in results.description])

        # ZLIB decompress the embeddings
        df["embedding"] = df["embedding"].apply(lambda x: np.frombuffer(zlib.decompress(x), dtype=np.int32).tolist())

        # Drop the `current` column
        df.drop(columns=["current"], inplace=True)

        return df
