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

        # set the unk. embedding
        self.unk_embedding = self.get_embedding(self.unknown_prompt, engine=self.embedding_model)

    @property
    def unk_embedding(self):
        return self._unk_embedding

    @unk_embedding.setter
    def unk_embedding(self, embedding):
        logger.info("Setting new UNK embedding...")
        self._unk_embedding = embedding
        return self._unk_embedding

    @lru_cache
    def get_embedding(self, query: str, engine: str):
        logger.info("generating embedding")
        return get_embedding(query, engine=engine)

    def check_sources_used(self, completion: Completion) -> bool:
        """Check to see if a response is relevant to the chatbot's knowledge or not.

        We assume we've prompt-engineered our bot to say a response is unrelated to the context if it isn't relevant.
        Here, we compare the embedding of the response to the embedding of the prompt-engineered "I don't know" embedding.

        set the unk_threshold to 0 to essentially turn off this feature.
        """

        engine: str = self.embedding_model
        unk_embedding: np.array = self.unk_embedding
        unk_threshold: float = self.unknown_threshold

        if completion.error:
            # considered not relevant if an error occured
            return False

        if completion.text == "":
            raise ValueError("Cannot compute embedding of an empty string.")

        response_embedding = get_embedding(
            completion.text,
            engine=engine,
        )
        score = cosine_similarity(response_embedding, unk_embedding)
        logger.info(f"UNK score: {score}")

        # Likely that the answer is meaningful, add the top sources
        return bool(score < unk_threshold)


def validator_factory(validator_cfg: dict) -> Validator:
    return Validator(validator_cfg=validator_cfg)
