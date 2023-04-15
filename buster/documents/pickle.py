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

    def __repr__(self):
        return "DocumentsPickle"

    def add(self, source: str, df: pd.DataFrame):
        """Write all documents from the dataframe into the db as a new version."""
        if source is not None:
            df["source"] = source

        df["current"] = 1

        if self.documents is not None:
            self.documents.loc[self.documents.source == source, "current"] = 0
            self.documents = pd.concat([self.documents, df])
        else:
            self.documents = df

        self.documents.to_pickle(self.filepath)

    def update_source(self, source: str, display_name: str = None, note: str = None):
        """Update the display name and/or note of a source. Also create the source if it does not exist."""
        print("If you need this function, please switch your backend to DocumentsDB.")
