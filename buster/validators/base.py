import logging
from abc import ABC, abstractmethod
from functools import lru_cache

import pandas as pd
from openai.embeddings_utils import cosine_similarity, get_embedding

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Validator(ABC):
    def __init__(
        self,
        embedding_model: str,
        unknown_threshold: float,
        use_reranking: bool,
        invalid_question_response: str = "This question is not relevant to my internal knowledge base.",
    ):
        self.embedding_model = embedding_model
        self.unknown_threshold = unknown_threshold
        self.use_reranking = use_reranking
        self.invalid_question_response = invalid_question_response

    @staticmethod
    @lru_cache
    def get_embedding(query: str, engine: str):
        """Currently supports OpenAI embeddings, override to add your own."""
        logger.info("generating embedding")
        return get_embedding(query, engine=engine)

    @abstractmethod
    def check_question_relevance(self, question: str) -> tuple[bool, str]:
        ...

    @abstractmethod
    def check_answer_relevance(self, answer: str) -> bool:
        ...

    def rerank_docs(self, answer: str, matched_documents: pd.DataFrame) -> pd.DataFrame:
        """Here we re-rank matched documents according to the answer provided by the llm.

        This score could be used to determine wether a document was actually relevant to generation.
        An extra column is added in-place for the similarity score.
        """
        if len(matched_documents) == 0:
            return matched_documents
        logger.info("Reranking documents based on answer similarity...")

        answer_embedding = self.get_embedding(
            answer,
            engine=self.embedding_model,
        )
        col = "similarity_to_answer"
        matched_documents[col] = matched_documents.embedding.apply(lambda x: cosine_similarity(x, answer_embedding))

        return matched_documents.sort_values(by=col, ascending=False)
