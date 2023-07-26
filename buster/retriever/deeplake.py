import logging

import pandas as pd

from buster.retriever.base import ALL_SOURCES, Retriever

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DeepLakeRetriever(Retriever):
    def __init__(self, path, **kwargs):
        from deeplake.core.vectorstore import VectorStore

        super().__init__(**kwargs)
        self.vector_store = VectorStore(
            path=path,
            read_only=True,
        )

    def get_documents(self, source: str = None):
        """Get all current documents from a given source."""
        raise NotImplementedError()

    def get_source_display_name(self, source: str) -> str:
        """Get the display name of a source.

        If source is None, returns all documents. If source does not exist, returns empty dataframe."""
        raise NotImplementedError()

    def get_topk_documents(self, query: str, source: str = None, top_k: int = None) -> pd.DataFrame:
        """Get the topk documents matching a user's query.

        If no matches are found, returns an empty dataframe."""

        query_embedding = self.get_embedding(query, engine=self.embedding_model)

        if source is not None:
            logger.info("Applying source {source} filter...")
            filter = {"metadata": {"source": source}}
        else:
            filter = None

        data = self.vector_store.search(
            k=top_k,
            embedding=query_embedding,
            exec_option="python",
            return_tensors="*",
            filter=filter,
        )
        # rename 'score' to 'similarity'
        data["similarity"] = data.pop("score")
        data["content"] = data.pop("text")

        matched_documents = pd.DataFrame(data)

        if len(matched_documents) == 0:
            logger.info("No matches found...")
            return pd.DataFrame()

        def extract_metadata(x, columns):
            """Returned metadata from deeplake is in a nested dict, extract it so that each attribute has its own column."""
            for col in columns:
                x[col] = x.metadata[col]
            return x

        matched_documents = matched_documents.apply(extract_metadata, columns=["source", "title", "url"], axis=1)
        matched_documents = matched_documents.drop(columns="metadata")

        return matched_documents
