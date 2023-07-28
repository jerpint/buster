import logging
from typing import Optional

import openai
import pandas as pd
from deeplake.core.vectorstore import VectorStore
from utils import zip_contents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def openai_embedding_function(texts, model="text-embedding-ada-002"):
    if isinstance(texts, str):
        texts = [texts]

    texts = [t.replace("\n", " ") for t in texts]
    return [data["embedding"] for data in openai.Embedding.create(input=texts, model=model)["data"]]


def extract_metadata(df: pd.DataFrame) -> dict:
    """extract the metadata from the dataframe in deeplake dict format"""
    metadata = df.apply(
        lambda x: {
            "url": x.url,
            "source": x.source,
            "title": x.title,
        },
        axis=1,
    ).to_list()
    return metadata


def read_csv(filename: str):
    """Assumes a pre-chunked csv file is provided with expected columns."""
    df = pd.read_csv(filename)
    for col in ["url", "source", "title", "content"]:
        assert col in df.columns
    return df

    # # read csv from disk
    # df = read_csv(csv_file)

    # # save the deeplake folder to a zip file
    # if zip_output_path is not None:


from base import DocumentsManager


class DeepLakeDocumentsManager(DocumentsManager):
    def __init__(self, vector_store_path, **vector_store_kwargs):
        self.vector_store_path = vector_store_path
        self.vector_store = VectorStore(
            path=vector_store_path,
            **vector_store_kwargs,
        )

    def add(self, df: pd.DataFrame):
        """Write all documents from the dataframe into the db as a new version.

        Each entry in the df is expected to have at least the following columns:
        ["url", "source", "title", "content"]
        """
        for col in ["url", "source", "title", "content"]:
            assert col in df.columns, "Check that all required columns are present."
        # extract the chunked text + metadata
        metadata = extract_metadata(df)
        chunked_text = df.content.to_list()

        # add the embeddings
        self.vector_store.add(
            text=chunked_text,
            embedding_function=openai_embedding_function,
            embedding_data=chunked_text,
            metadata=metadata,
        )

    def to_zip(output_path: str = "."):
        """Zip the contents of the vector_store_path folder to a .zip file in output_path."""
        logger.info(f"Compressing {vector_store_path}...")
        zip_file_path = zip_contents(input_path=vector_store_path, output_path=output_path)
        logger.info(f"Compressed {vector_store_path} to {zip_file_path}.")
        return zip_file_path


if __name__ == "__main__":
    vector_store_path = "deeplake_store"
    csv_file = "data/chunks_preprocessed.csv"
    overwrite = True

    docs_manager = DeepLakeDocumentsManager(vector_store_path=vector_store_path, overwrite=True)
    zip_file = docs_manager.to_zip()