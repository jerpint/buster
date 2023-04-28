import logging
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from openai.embeddings_utils import cosine_similarity, get_embedding

from buster.completers import completer_factory
from buster.completers.base import Completion
from buster.formatters.documents import document_formatter_factory
from buster.formatters.prompts import SystemPromptFormatter, prompt_formatter_factory
from buster.retriever import Retriever
from buster.tokenizers import tokenizer_factory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass(slots=True)
class Response:
    completion: Completion
    is_relevant: bool
    user_input: str
    matched_documents: pd.DataFrame | None = None


@dataclass
class BusterConfig:
    """Configuration object for a chatbot."""

    embedding_model: str = "text-embedding-ada-002"
    unknown_threshold: float = 0.85
    unknown_prompt: str = "I Don't know how to answer your question."
    document_source: str = ""
    tokenizer_cfg: dict = field(
        default_factory=lambda: {
            "model_name": "gpt-3.5-turbo",
        }
    )
    retriever_cfg: dict = field(
        default_factory=lambda: {
            "top_k": 3,
            "thresh": 0.7,
        }
    )
    prompt_cfg: dict = field(
        default_factory=lambda: {
            "max_tokens": 2000,
            "text_before_documents": "You are a chatbot answering questions.\n",
            "text_before_prompt": "Answer the following question:\n",
        }
    )
    completion_cfg: dict = field(
        default_factory=lambda: {
            "name": "ChatGPT",
            "completion_kwargs": {
                "engine": "gpt-3.5-turbo",
                "max_tokens": 200,
                "temperature": None,
                "top_p": None,
                "frequency_penalty": 1,
                "presence_penalty": 1,
            },
        }
    )


class Buster:
    def __init__(self, cfg: BusterConfig, retriever: Retriever):
        self._unk_embedding = None
        self.update_cfg(cfg)

        self.retriever = retriever

    @property
    def unk_embedding(self):
        return self._unk_embedding

    @unk_embedding.setter
    def unk_embedding(self, embedding):
        logger.info("Setting new UNK embedding...")
        self._unk_embedding = embedding
        return self._unk_embedding

    def update_cfg(self, cfg: BusterConfig):
        """Every time we set a new config, we update the things that need to be updated."""
        logger.info(f"Updating config to {cfg.document_source}:\n{cfg}")
        self._cfg = cfg
        self.embedding_model = cfg.embedding_model
        self.unknown_threshold = cfg.unknown_threshold
        self.unknown_prompt = cfg.unknown_prompt
        self.document_source = cfg.document_source

        self.retriever_cfg = cfg.retriever_cfg
        self.completion_cfg = cfg.completion_cfg
        self.prompt_cfg = cfg.prompt_cfg
        self.tokenizer_cfg = cfg.tokenizer_cfg

        # set the unk. embedding
        self.unk_embedding = self.get_embedding(self.unknown_prompt, engine=self.embedding_model)

        # update completer and formatter cfg
        self.tokenizer = tokenizer_factory(self.tokenizer_cfg)
        self.completer = completer_factory(self.completion_cfg)
        self.documents_formatter = document_formatter_factory(
            tokenizer=self.tokenizer,
            max_tokens=self.retriever_cfg["max_tokens"]
            # TODO: move max_tokens from retriever_cfg to somewhere more logical
        )
        self.prompt_formatter = prompt_formatter_factory(tokenizer=self.tokenizer, prompt_cfg=self.prompt_cfg)

        logger.info(f"Config Updated.")

    @lru_cache
    def get_embedding(self, query: str, engine: str):
        logger.info("generating embedding")
        return get_embedding(query, engine=engine)

    def rank_documents(
        self,
        query: str,
        top_k: float,
        thresh: float,
        engine: str,
        source: str,
    ) -> pd.DataFrame:
        """
        Compare the question to the series of documents and return the best matching documents.
        """

        query_embedding = self.get_embedding(
            query,
            engine=engine,
        )
        matched_documents = self.retriever.retrieve(query_embedding, top_k=top_k, source=source)

        # log matched_documents to the console
        logger.info(f"matched documents before thresh: {matched_documents}")

        # filter out matched_documents using a threshold
        matched_documents = matched_documents[matched_documents.similarity > thresh]
        logger.info(f"matched documents after thresh: {matched_documents}")

        return matched_documents

    def check_response_relevance(
        self, completion_text: str, engine: str, unk_embedding: np.array, unk_threshold: float
    ) -> bool:
        """Check to see if a response is relevant to the chatbot's knowledge or not.

        We assume we've prompt-engineered our bot to say a response is unrelated to the context if it isn't relevant.
        Here, we compare the embedding of the response to the embedding of the prompt-engineered "I don't know" embedding.

        set the unk_threshold to 0 to essentially turn off this feature.
        """
        response_embedding = self.get_embedding(
            completion_text,
            engine=engine,
        )
        score = cosine_similarity(response_embedding, unk_embedding)
        logger.info(f"UNK score: {score}")

        # Likely that the answer is meaningful, add the top sources
        return bool(score < unk_threshold)

    def process_input(self, user_input: str) -> Response:
        """
        Main function to process the input question and generate a formatted output.
        """

        logger.info(f"User Input:\n{user_input}")

        # We make sure there is always a newline at the end of the question to avoid completing the question.
        if not user_input.endswith("\n"):
            user_input += "\n"

        matched_documents = self.rank_documents(
            query=user_input,
            top_k=self.retriever_cfg["top_k"],
            thresh=self.retriever_cfg["thresh"],
            engine=self.embedding_model,
            source=self.document_source,
        )

        if len(matched_documents) == 0:
            logger.warning("No documents found...")
            completion = Completion(text="No documents found.")
            matched_documents = pd.DataFrame(columns=matched_documents.columns)
            response = Response(
                completion=completion, matched_documents=matched_documents, is_relevant=False, user_input=user_input
            )
            return response

        # format the matched documents, (will truncate them if too long)
        documents_str, matched_documents = self.documents_formatter.format(matched_documents)

        # prepare the prompt
        system_prompt = self.prompt_formatter.format(documents_str)
        completion: Completion = self.completer.generate_response(user_input=user_input, system_prompt=system_prompt)
        logger.info(f"GPT Response:\n{completion.text}")

        # check for relevance
        is_relevant = self.check_response_relevance(
            completion_text=completion.text,
            engine=self.embedding_model,
            unk_embedding=self.unk_embedding,
            unk_threshold=self.unknown_threshold,
        )
        if not is_relevant:
            matched_documents = pd.DataFrame(columns=matched_documents.columns)
            # answer generated was the chatbot saying it doesn't know how to answer
        # uncomment override completion with unknown prompt
        # completion = Completion(text=self.unknown_prompt)

        response = Response(
            completion=completion, matched_documents=matched_documents, is_relevant=is_relevant, user_input=user_input
        )
        return response
