import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from openai.embeddings_utils import cosine_similarity, get_embedding

from buster.completers.base import Completion

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Validator:
    def __init__(self, validator_cfg):
        self.cfg = validator_cfg

        self.embedding_model = self.cfg["embedding_model"]
        self.unknown_threshold = self.cfg["unknown_threshold"]
        self.unknown_prompt = self.cfg["unknown_prompt"]
        self.use_reranking = self.cfg["use_reranking"]


    @lru_cache
    def get_embedding(self, query: str, engine: str):
        logger.info("generating embedding")
        return get_embedding(query, engine=engine)

    def check_response_relevant(self, completion: Completion) -> bool:
        """Check to see if a response is relevant to the chatbot's knowledge or not.

        We assume we've prompt-engineered our bot to say a response is unrelated to the context if it isn't relevant.
        Here, we compare the embedding of the response to the embedding of the prompt-engineered "I don't know" embedding.

        unk_threshold can be a value between [-1,1]. Usually, 0.85 is a good value.
        """

        if completion.error:
            # considered not relevant if an error occured
            return False

        if completion.text == "":
            raise ValueError("Cannot compute embedding of an empty string.")

        unknown_embedding = self.get_embedding(
            self.unknown_prompt,
            engine=self.embedding_model,
        )

        answer_embedding = self.get_embedding(
            completion.text,
            engine=self.embedding_model,
        )
        unknown_similarity_score = cosine_similarity(answer_embedding, unknown_embedding)
        logger.info(f"{unknown_similarity_score=}")

        # Likely that the answer is meaningful, add the top sources
        return bool(unknown_similarity_score < self.unknown_threshold)

    def rerank_docs(self, completion: Completion, matched_documents: pd.DataFrame):
        """Here we re-rank matched documents according to the answer provided by the llm.

        This score could be used to determine wether a document was actually relevant to generation.
        An extra column is added in-place for the similarity score.
        """
        logger.info("Reranking documents based on answer similarity...")
        answer_embedding = self.get_embedding(
            completion.text,
            engine=self.embedding_model,
        )
        col = "similarity_to_answer"
        matched_documents[col] = matched_documents.embedding.apply(
            lambda x: cosine_similarity(x, answer_embedding) * 100
        )

        return matched_documents.sort_values(by=col, ascending=False)


def validator_factory(validator_cfg: dict) -> Validator:
    return Validator(validator_cfg=validator_cfg)
