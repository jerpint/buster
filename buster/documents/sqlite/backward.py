"""Used to import existing DB as a new DB."""

import argparse
import itertools
import sqlite3
from typing import Iterable, NamedTuple

import numpy as np

import buster.documents.sqlite.documents as dest
from buster.documents.sqlite import DocumentsDB

IMPORT_QUERY = (
    r"""SELECT source, url, title, content FROM documents WHERE current = 1 ORDER BY source, url, title, id"""
)
CHUNK_QUERY = r"""SELECT source, url, title, content, n_tokens, embedding FROM documents WHERE current = 1 ORDER BY source, url, id"""


class Document(NamedTuple):
    """Document from the original db."""

    source: str
    url: str
    title: str
    content: str


class Section(NamedTuple):
    """Reassemble section from the original db."""

    url: str
    title: str
    content: str


class Chunk(NamedTuple):
    """Chunk from the original db."""

    source: str
    url: str
    title: str
    content: str
    n_tokens: int
    embedding: np.ndarray


def get_documents(conn: sqlite3.Connection) -> Iterable[tuple[str, Iterable[Section]]]:
    """Reassemble documents from the source db's chunks."""
    documents = (Document(*row) for row in conn.execute(IMPORT_QUERY))
    by_sources = itertools.groupby(documents, lambda doc: doc.source)
    for source, documents in by_sources:
        documents = itertools.groupby(documents, lambda doc: (doc.url, doc.title))
        sections = (
            Section(url, title, "".join(chunk.content for chunk in chunks)) for (url, title), chunks in documents
        )
        yield source, sections


def get_max_size(conn: sqlite3.Connection) -> int:
    """Get the maximum chunk size from the source db."""
    sizes = (size for size, in conn.execute("select max(length(content)) FROM documents"))
    (size,) = sizes
    return size


def get_chunks(conn: sqlite3.Connection) -> Iterable[tuple[str, Iterable[Iterable[dest.Chunk]]]]:
    """Retrieve chunks from the source db."""
    chunks = (Chunk(*row) for row in conn.execute(CHUNK_QUERY))
    by_sources = itertools.groupby(chunks, lambda chunk: chunk.source)
    for source, chunks in by_sources:
        by_section = itertools.groupby(chunks, lambda chunk: (chunk.url, chunk.title))

        sections = (
            (dest.Chunk(chunk.content, chunk.n_tokens, chunk.embedding) for chunk in chunks) for _, chunks in by_section
        )

        yield source, sections


def main():
    """Import the source db into the destination db."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--size", type=int, default=2000)
    args = parser.parse_args()
    org = sqlite3.connect(args.source)
    db = DocumentsDB(args.destination)

    for source, content in get_documents(org):
        # sid, vid = db.start_version(source)
        sections = (dest.Section(section.title, section.url, section.content) for section in content)
        db.add_parse(source, sections)

    size = max(args.size, get_max_size(org))
    for source, chunks in get_chunks(org):
        sid, vid = db.get_current_version(source)
        db.add_chunking(sid, vid, size, chunks)
    db.conn.commit()

    return


if __name__ == "__main__":
    main()
