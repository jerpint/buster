import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from buster.retriever.base import ALL_SOURCES, Retriever

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract_metadata(x: pd.DataFrame, columns) -> pd.DataFrame:
    """Returned metadata from deeplake is in a nested dict, extract it so that each attribute has its own column."""
    for col in columns:
        x[col] = x.metadata[col]
    return x


def data_dict_to_df(data: dict):
    # rename 'score' to 'similarity'
    data["similarity"] = data.pop("score")
    data["content"] = data.pop("text")

    matched_documents = pd.DataFrame(data)

    if len(matched_documents) == 0:
        logger.info("No matches found...")
        return pd.DataFrame()

    matched_documents = matched_documents.apply(extract_metadata, columns=["source", "title", "url"], axis=1)
    matched_documents = matched_documents.drop(columns="metadata")

    return matched_documents


def build_tql_query(embedding, sources=None, top_k: int = 3):
    # Initialize the where_clause to an empty string.
    where_clause = ""

    embedding_string = ",".join([str(item) for item in embedding])

    # If sources is provided and it's not empty, build the where clause.
    if sources:
        conditions = [f"contains(metadata['source'], '{source}')" for source in sources]
        where_clause = "where " + " or ".join(conditions)

    # Construct the entire query
    query = f"""
select * from (
    select embedding, text, metadata, cosine_similarity(embedding, ARRAY[{embedding_string}]) as score
    {where_clause}
)
order by score desc limit {top_k}
"""
    return query


class DeepLakeRetriever(Retriever):
    def __init__(
        self,
        path,
        exec_option: str = "python",
        use_tql: bool = False,
        deep_memory: bool = False,
        activeloop_token: str = None,
        **kwargs,
    ):
        from deeplake.core.vectorstore import VectorStore

        super().__init__(**kwargs)
        if activeloop_token is None:
            logger.warning(
                """
                No activeloop token detected, enterprise features will not be available.
                You can set it using: export ACTIVELOOP_TOKEN=...
                """
            )
        self.use_tql = use_tql
        self.exec_option = exec_option
        self.deep_memory = deep_memory
        self.vector_store = VectorStore(
            path=path,
            read_only=True,
            token=activeloop_token,
            exec_option=exec_option,
        )

    def get_documents(self, sources: Optional[list[str]] = None):
        """Get all current documents from a given source."""
        k = len(self.vector_store)

        # currently this is the only way to retrieve all embeddings in deeplake
        # generate a dummy embedding and specify top-k equals the length of the vector store.
        embedding_dim = self.vector_store.tensors()["embedding"].shape[1]
        dummy_embedding = np.random.random(embedding_dim)

        return self.get_topk_documents(query=None, embedding=dummy_embedding, top_k=k, sources=sources)

    def get_source_display_name(self, source: str) -> str:
        """Get the display name of a source.

        If source is None, returns all documents. If source does not exist, returns empty dataframe."""
        raise NotImplementedError()

    def get_topk_documents(
        self,
        query: str = None,
        embedding: np.array = None,
        sources: Optional[list[str]] = None,
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

        if self.use_tql:
            assert self.exec_option == "compute_engine", "cant use tql without compute_engine"
            tql_query = build_tql_query(query_embedding, sources=sources, top_k=top_k)
            data = self.vector_store.search(query=tql_query, deep_memory=self.deep_memory)
        else:
            # build the filter clause
            if sources:

                def filter(x):
                    return x["metadata"].data()["value"]["source"] in sources

            else:
                filter = None

            data = self.vector_store.search(
                k=top_k,
                embedding=query_embedding,
                exec_option=self.exec_option,
                return_tensors=return_tensors,
                filter=filter,
            )

        matched_documents = data_dict_to_df(data)
        return matched_documents
