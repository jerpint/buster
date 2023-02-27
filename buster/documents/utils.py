import os
from typing import Type

from buster.documents.base import DocumentsManager
from buster.documents.pickle import DocumentsPickle
from buster.documents.sqlite import DocumentsDB

PICKLE_EXTENSIONS = [".gz", ".bz2", ".zip", ".xz", ".zst", ".tar", ".tar.gz", ".tar.xz", ".tar.bz2"]


def get_file_extension(filepath: str) -> str:
    return os.path.splitext(filepath)[1]


def get_documents_manager_from_extension(filepath: str) -> Type[DocumentsManager]:
    ext = get_file_extension(filepath)

    if ext in PICKLE_EXTENSIONS:
        return DocumentsPickle
    elif ext == ".db":
        return DocumentsDB
    else:
        raise ValueError(f"Unsupported format: {ext}.")
