import os
import urllib.request
from typing import Type

from buster.documents import DocumentsDB, DocumentsManager
from buster.retriever import Retriever, SQLiteRetriever

def get_file_extension(filepath: str) -> str:
    return os.path.splitext(filepath)[1]


def download_db(db_url: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, "documents.db")
    if not os.path.exists(fname):
        print(f"Downloading db file from {db_url} to {fname}...")
        urllib.request.urlretrieve(db_url, fname)
        print("Downloaded.")
    else:
        print("File already exists. Skipping.")
    return fname


def get_documents_manager_from_extension(filepath: str) -> Type[DocumentsManager]:
    ext = get_file_extension(filepath)

    if ext == ".db":
        return DocumentsDB
    else:
        raise ValueError(f"Unsupported format: {ext}.")


def get_retriever_from_extension(filepath: str) -> Type[Retriever]:
    ext = get_file_extension(filepath)

    if ext == ".db":
        return SQLiteRetriever
    else:
        raise ValueError(f"Unsupported format: {ext}.")
