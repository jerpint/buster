import sqlite3
from typing import Iterable, NamedTuple
import warnings
import zlib

import numpy as np
import pandas as pd

import buster.db.schema as schema


class Section(NamedTuple):
    title: str
    url: str
    content: str
    parent: int | None = None
    type: str = "section"


class Chunk(NamedTuple):
    content: str
    n_tokens: int
    emb: np.ndarray


class DocumentsDB:
    """Simple SQLite database for storing documents and questions/answers.

    The database is just a file on disk. It can store documents from different sources, and it can store multiple versions of the same document (e.g. if the document is updated).
    Questions/answers refer to the version of the document that was used at the time.

    Example:
        >>> db = DocumentsDB("/path/to/the/db.db")
        >>> db.write_documents("source", df)  # df is a DataFrame containing the documents from a given source, obtained e.g. by using buster.docparser.generate_embeddings
        >>> df = db.get_documents("source")
    """

    def __init__(self, db_path: sqlite3.Connection | str):
        if isinstance(db_path, str):
            self.db_path = db_path
            self.conn = sqlite3.connect(db_path)
        else:
            self.db_path = None
            self.conn = db_path
        self.cursor = self.conn.cursor()
        schema.initialize_db(self.conn)
        schema.setup_db(self.conn)

    def __del__(self):
        if self.db_path is not None:
            self.conn.close()

    def get_current_version(self, source: str) -> tuple[int, int]:
        cur = self.conn.execute("SELECT source, version FROM latest_version WHERE name = ?", (source,))
        row = cur.fetchone()
        if row is None:
            raise KeyError(f'"{source}" is not a known source')
        sid, vid = row
        return sid, vid

    def get_source(self, source: str) -> int:
        cur = self.conn.execute("SELECT id FROM sources WHERE name = ?", (source,))
        row = cur.fetchone()
        if row is not None:
            (sid,) = row
        else:
            cur = self.conn.execute("INSERT INTO sources (name) VALUES (?)", (source,))
            cur = self.conn.execute("SELECT id FROM sources WHERE name = ?", (source,))
            row = cur.fetchone()
            (sid,) = row

        return sid

    def start_version(self, source: str) -> tuple[int, int]:
        cur = self.conn.execute("SELECT source, version FROM latest_version WHERE name = ?", (source,))
        row = cur.fetchone()
        if row is None:
            sid = self.get_source(source)
            vid = 0
        else:
            sid, vid = row
            vid = vid + 1
        self.conn.execute("INSERT INTO versions (source, version) VALUES (?, ?)", (sid, vid))
        return sid, vid

    def add_sections(self, sid: int, vid: int, sections: Iterable[Section]):
        values = (
            (sid, vid, ind, section.title, section.url, section.content, section.parent, section.type)
            for ind, section in enumerate(sections)
        )
        self.conn.executemany(
            "INSERT INTO sections "
            "(source, version, section, title, url, content, parent, type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        return

    def add_chunking(self, sid: int, vid: int, size: int, overlap: int = 0, strategy: str = "simple") -> int:
        self.conn.execute(
            "INSERT INTO chunkings (size, overlap, strategy, source, version) VALUES (?, ?, ?, ?, ?)",
            (size, overlap, strategy, sid, vid),
        )
        cur = self.conn.execute(
            "SELECT chunking FROM chunkings "
            "WHERE size = ? AND overlap = ? AND strategy = ? AND source = ? AND version = ?",
            (size, overlap, strategy, sid, vid),
        )
        (id,) = (id for id, in cur)
        return id

    def add_chunks(self, sid: int, vid: int, cid: int, sections: Iterable[Iterable[Chunk]]):
        chunks = ((ind, jnd, chunk) for ind, section in enumerate(sections) for jnd, chunk in enumerate(section))
        values = ((sid, vid, ind, cid, jnd, chunk.content, chunk.n_tokens, chunk.emb) for ind, jnd, chunk in chunks)
        self.conn.executemany(
            "INSERT INTO chunks "
            "(source, version, section, chunking, sequence, content, n_tokens, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        return

    def write_documents(self, source: str, df: pd.DataFrame):
        """Write all documents from the dataframe into the db. All previous documents from that source will be set to `current = 0`."""
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

            # Check that the embeddings are float32
            if not df["embedding"].iloc[0].dtype == np.float32:
                warnings.warn(
                    f"Embeddings are not float32, converting them to float32 from {df['embedding'].iloc[0].dtype}.",
                    RuntimeWarning,
                )
                df["embedding"] = df["embedding"].apply(lambda x: x.astype(np.float32))

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
        results = self.cursor.execute("SELECT * FROM documents WHERE source = ?", (source,))
        rows = results.fetchall()

        # Convert the results to a pandas DataFrame
        df = pd.DataFrame(rows, columns=[description[0] for description in results.description])
        return df
