import sqlite3
import zlib

import numpy as np

SOURCE_TABLE = r"""CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    note TEXT,
    UNIQUE(name)
)"""


VERSION_TABLE = r"""CREATE TABLE IF NOT EXISTS versions (
    source INTEGER,
    version INTEGER,
    parser TEXT,
    note TEXT,
    PRIMARY KEY (version, source, parser)
    FOREIGN KEY (source) REFERENCES sources (id)
)"""


CHUNKING_TABLE = r"""CREATE TABLE IF NOT EXISTS chunkings (
    chunking INTEGER PRIMARY KEY AUTOINCREMENT,
    size INTEGER,
    overlap INTEGER,
    strategy TEXT,
    chunker TEXT,
    source INTEGER,
    version INTEGER,
    UNIQUE (size, overlap, strategy, chunker, source, version),
    FOREIGN KEY (source, version) REFERENCES versions (source, version)
)"""


SECTION_TABLE = r"""CREATE TABLE IF NOT EXISTS sections (
    source INTEGER,
    version INTEGER,
    section INTEGER,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    content TEXT NOT NULL,
    parent INTEGER,
    type TEXT,
    PRIMARY KEY (version, source, section),
    FOREIGN KEY (source) REFERENCES versions (source),
    FOREIGN KEY (version) REFERENCES versions (version)
)"""


CHUNK_TABLE = r"""CREATE TABLE IF NOT EXISTS chunks (
    source INTEGER,
    version INTEGER,
    section INTEGER,
    chunking INTEGER,
    sequence INTEGER,
    content TEXT NOT NULL,
    n_tokens INTEGER,
    embedding VECTOR,
    PRIMARY KEY (source, version, section, chunking, sequence),
    FOREIGN KEY (source, version, section) REFERENCES sections (source, version, section),
    FOREIGN KEY (source, version, chunking) REFERENCES chunkings (source, version, chunking)
)"""


VERSION_VIEW = r"""CREATE VIEW IF NOT EXISTS latest_version (
    name, source, version) AS
    SELECT sources.name, versions.source, max(versions.version)
    FROM sources INNER JOIN versions on sources.id = versions.source
    GROUP BY sources.id
"""

CHUNKING_VIEW = r"""CREATE VIEW IF NOT EXISTS latest_chunking (
    name, source, version, chunking) AS
    SELECT name, source, version, max(chunking) FROM
    chunkings INNER JOIN latest_version USING (source, version)
    GROUP by source, version
"""

DOCUMENT_VIEW = r"""CREATE VIEW IF NOT EXISTS documents (
    source, title, url, content, n_tokens, embedding)
    AS SELECT latest_chunking.name, sections.title, sections.url,
    chunks.content, chunks.n_tokens, chunks.embedding
    FROM chunks INNER JOIN sections USING (source, version, section)
    INNER JOIN latest_chunking USING (source, version, chunking)
"""


INIT_STATEMENTS = [
    SOURCE_TABLE,
    VERSION_TABLE,
    CHUNKING_TABLE,
    SECTION_TABLE,
    CHUNK_TABLE,
    VERSION_VIEW,
    CHUNKING_VIEW,
    DOCUMENT_VIEW,
]


def initialize_db(connection: sqlite3.Connection):
    for statement in INIT_STATEMENTS:
        try:
            connection.execute(statement)
        except sqlite3.Error as error:
            connection.rollback()
            raise
    connection.commit()
    return connection


def adapt_vector(vector: np.ndarray) -> bytes:
    return sqlite3.Binary(zlib.compress(vector.astype(np.float32).tobytes()))


def convert_vector(buffer: bytes) -> np.ndarray:
    return np.frombuffer(zlib.decompress(buffer), dtype=np.float32)


def cosine_similarity(a: bytes, b: bytes) -> float:
    a = convert_vector(a)
    b = convert_vector(b)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dopt = 0.5 * np.dot(a, b) + 0.5
    return float(dopt)


def setup_db(connection: sqlite3.Connection):
    sqlite3.register_adapter(np.ndarray, adapt_vector)
    sqlite3.register_converter("vector", convert_vector)
    connection.create_function("sim", 2, cosine_similarity, deterministic=True)
