import os

import pandas as pd

from buster.documents.base import DocumentsManager


class DocumentsPickle(DocumentsManager):
    def __init__(self, filepath: str):
        self.filepath = filepath

        if os.path.exists(filepath):
            self.documents = pd.read_pickle(filepath)
        else:
            self.documents = None

    def add(self, source: str, df: pd.DataFrame):
        if source is not None:
            df["source"] = source

        df["current"] = 1

        if self.documents is not None:
            self.documents.loc[self.documents.source == source, "current"] = 0
            self.documents = pd.concat([self.documents, df])
        else:
            self.documents = df

        self.documents.to_pickle(self.filepath)

    def get_documents(self, source: str) -> pd.DataFrame:
        if self.documents is None:
            raise FileNotFoundError(f"No documents found at {self.filepath}. Are you sure this is the correct path?")

        documents = self.documents.copy()
        if "current" in documents.columns:
            documents = documents[documents.current == 1]

            # Drop the `current` column
            documents.drop(columns=["current"], inplace=True)

        if source is not None and "source" in documents.columns:
            documents = documents[documents.source == source]

        return documents
