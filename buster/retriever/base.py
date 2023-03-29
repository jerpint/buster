from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
from openai.embeddings_utils import cosine_similarity


@dataclass
class Retriever(ABC):
    @abstractmethod
    def get_documents(self, source: str) -> pd.DataFrame:
        ...

    def retrieve(self, query_embedding: list[float], top_k: int, source: str = None) -> pd.DataFrame:
        documents = self.get_documents(source)

        documents["similarity"] = documents.embedding.apply(lambda x: cosine_similarity(x, query_embedding))

        # sort the matched_documents by score
        matched_documents = documents.sort_values("similarity", ascending=False)

        # limit search to top_k matched_documents.
        top_k = len(matched_documents) if top_k == -1 else top_k
        matched_documents = matched_documents.head(top_k)

        return matched_documents