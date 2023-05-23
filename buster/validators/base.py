import logging
from functools import lru_cache

import pandas as pd
from openai.embeddings_utils import cosine_similarity, get_embedding

from buster.completers.base import Completion

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Validator:
    def __init__(self, embedding_model: str, unknown_threshold: float, unknown_prompt: str, use_reranking: bool):
        self.embedding_model = embedding_model
        self.unknown_threshold = unknown_threshold
        self.unknown_prompt = unknown_prompt
        self.use_reranking = use_reranking

    @staticmethod
    @lru_cache
    def get_embedding(query: str, engine: str):
        logger.info("generating embedding")
        return get_embedding(query, engine=engine)

    def check_answer_relevance(self, answer: str, unknown_prompt: str = None) -> bool:
        """Check to see if a generated answer is relevant to the chatbot's knowledge or not.

        We assume we've prompt-engineered our bot to say a response is unrelated to the context if it isn't relevant.
        Here, we compare the embedding of the response to the embedding of the prompt-engineered "I don't know" embedding.

        unk_threshold can be a value between [-1,1]. Usually, 0.85 is a good value.
        """
        logger.info("Checking for answer relevance...")

        if answer == "" or unknown_prompt == "":
            raise ValueError("Cannot compute embedding of an empty string.")

        if unknown_prompt is None:
            unknown_prompt = self.unknown_prompt

        unknown_embedding = self.get_embedding(
            unknown_prompt,
            engine=self.embedding_model,
        )

        answer_embedding = self.get_embedding(
            answer,
            engine=self.embedding_model,
        )
        unknown_similarity_score = cosine_similarity(answer_embedding, unknown_embedding)
        logger.info(f"{unknown_similarity_score=}")

        # Likely that the answer is meaningful, add the top sources
        return bool(unknown_similarity_score < self.unknown_threshold)

    def rerank_docs(self, answer: str, matched_documents: pd.DataFrame) -> pd.DataFrame:
        """Here we re-rank matched documents according to the answer provided by the llm.

        This score could be used to determine wether a document was actually relevant to generation.
        An extra column is added in-place for the similarity score.
        """
        logger.info("Reranking documents based on answer similarity...")

        answer_embedding = self.get_embedding(
            answer,
            engine=self.embedding_model,
        )
        col = "similarity_to_answer"
        matched_documents[col] = matched_documents.embedding.apply(
            lambda x: cosine_similarity(x, answer_embedding) * 100
        )

        return matched_documents.sort_values(by=col, ascending=False)

    def validate(self, completion: Completion) -> Completion:
        if completion.error:
            completion.answer_relevant = False
        elif len(completion.matched_documents) == 0:
            completion.answer_relevant = False
        else:
            completion.answer_relevant = self.check_answer_relevance(completion.text)

        completion.matched_documents = self.rerank_docs(completion.text, completion.matched_documents)

        return completion


def validator_factory(validator_cfg: dict) -> Validator:
    return Validator(validator_cfg=validator_cfg)
