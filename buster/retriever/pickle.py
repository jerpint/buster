import pandas as pd

from buster.retriever.base import Retriever


class PickleRetriever(Retriever):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.documents = pd.read_pickle(filepath)

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
