import itertools
import sqlite3
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd

import buster.documents.sqlite.schema as schema
from buster.documents.base import DocumentsManager


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


class DocumentsDB(DocumentsManager):
    """Simple SQLite database for storing documents and questions/answers.

    The database is just a file on disk. It can store documents from different sources, and it can store multiple versions of the same document (e.g. if the document is updated).
    Questions/answers refer to the version of the document that was used at the time.

    Example:
        >>> db = DocumentsDB("/path/to/the/db.db")
        >>> db.add("source", df)  # df is a DataFrame containing the documents from a given source, obtained e.g. by using buster.docparser.generate_embeddings
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

    def get_current_version(self, source: str) -> tuple[int, int]:
        """Get the current version of a source."""
        cur = self.conn.execute("SELECT source, version FROM latest_version WHERE name = ?", (source,))
        row = cur.fetchone()
        if row is None:
            raise KeyError(f'"{source}" is not a known source')
        sid, vid = row
        return sid, vid

    def get_source(self, source: str) -> int:
        """Get the id of a source."""
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

    def new_version(self, source: str) -> tuple[int, int]:
        """Create a new version for a source."""
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

    def add_parse(self, source: str, sections: Iterable[Section]) -> tuple[int, int]:
        """Create a new version of a source filled with parsed sections."""
        sid, vid = self.new_version(source)
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
        return sid, vid

    def new_chunking(self, sid: int, vid: int, size: int, overlap: int = 0, strategy: str = "simple") -> int:
        """Create a new chunking for a source."""
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

    def add_chunking(self, sid: int, vid: int, size: int, sections: Iterable[Iterable[Chunk]]) -> int:
        """Create a new chunking for a source, filled with chunks organized by section."""
        cid = self.new_chunking(sid, vid, size)
        chunks = ((ind, jnd, chunk) for ind, section in enumerate(sections) for jnd, chunk in enumerate(section))
        values = ((sid, vid, ind, cid, jnd, chunk.content, chunk.n_tokens, chunk.emb) for ind, jnd, chunk in chunks)
        self.conn.executemany(
            "INSERT INTO chunks "
            "(source, version, section, chunking, sequence, content, n_tokens, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        return cid

    def add(self, source: str, df: pd.DataFrame):
        """Write all documents from the dataframe into the db as a new version."""
        data = sorted(df.itertuples(), key=lambda chunk: (chunk.url, chunk.title))
        sections = []
        size = 0
        for (url, title), chunks in itertools.groupby(data, lambda chunk: (chunk.url, chunk.title)):
            chunks = [Chunk(chunk.content, chunk.n_tokens, chunk.embedding) for chunk in chunks]
            size = max(size, max(len(chunk.content) for chunk in chunks))
            content = "".join(chunk.content for chunk in chunks)
            sections.append((Section(title, url, content), chunks))

        sid, vid = self.add_parse(source, (section for section, _ in sections))
        self.add_chunking(sid, vid, size, (chunks for _, chunks in sections))
        self.conn.commit()

    def get_documents(self, source: str) -> pd.DataFrame:
        """Get all current documents from a given source."""
        # Execute the SQL statement and fetch the results
        results = self.conn.execute("SELECT * FROM documents WHERE source = ?", (source,))
        rows = results.fetchall()

        # Convert the results to a pandas DataFrame
        df = pd.DataFrame(rows, columns=[description[0] for description in results.description])
        return df
