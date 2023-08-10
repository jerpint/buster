import logging

import numpy as np
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
        k = len(self.vector_store)

        # currently this is the only way to retrieve all embeddings in deeplake
        # generate a dummy embedding and specify top-k equals the length of the vector store.
        embedding_dim = self.vector_store.tensors()["embedding"].shape[1]
        dummy_embedding = np.random.random(embedding_dim)

        return self.get_topk_documents(query=None, embedding=dummy_embedding, top_k=k, source=source)

    def get_source_display_name(self, source: str) -> str:
        """Get the display name of a source.

        If source is None, returns all documents. If source does not exist, returns empty dataframe."""
        raise NotImplementedError()

    def get_topk_documents(
        self,
        query: str = None,
        embedding: np.array = None,
        source: str = None,
        top_k: int = None,
        return_tensors: str = "*",
    ) -> pd.DataFrame:
        """Get the topk documents matching a user's query.

        If no matches are found, returns an empty dataframe."""

        if query is not None:
            query_embedding = self.get_embedding(query, model=self.embedding_model)
        elif embedding is not None:
            query_embedding = embedding
        else:
            raise ValueError("must provide either a query or an embedding")

        if source is not None:
            logger.info(f"Applying source {source} filter...")
            filter = {"metadata": {"source": source}}
        else:
            filter = None

        data = self.vector_store.search(
            k=top_k,
            embedding=query_embedding,
            exec_option="python",
            return_tensors=return_tensors,
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
