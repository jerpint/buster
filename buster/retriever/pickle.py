import pandas as pd

from buster.retriever.base import ALL_SOURCES, Retriever


class PickleRetriever(Retriever):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.documents = pd.read_pickle(filepath)

    def get_documents(self, source: str) -> pd.DataFrame:
        """Get all current documents from a given source."""
        if self.documents is None:
            raise FileNotFoundError(f"No documents found at {self.filepath}. Are you sure this is the correct path?")

        documents = self.documents.copy()
        # The `current` column exists when multiple versions of a document exist
        if "current" in documents.columns:
            documents = documents[documents.current == 1]

            # Drop the `current` column
            documents.drop(columns=["current"], inplace=True)

        if source not in [None, ""] and "source" in documents.columns:
            documents = documents[documents.source == source]

        return documents

    def get_source_display_name(self, source: str) -> str:
        """Get the display name of a source."""
        if source is None:
            return ALL_SOURCES
        else:
            return source
