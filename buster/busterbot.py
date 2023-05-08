import logging
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from openai.embeddings_utils import get_embedding

from buster.completers import completer_factory
from buster.completers.base import Completion
from buster.formatters.documents import document_formatter_factory
from buster.formatters.prompts import prompt_formatter_factory
from buster.retriever import Retriever
from buster.tokenizers import tokenizer_factory
from buster.validators.base import Validator, validator_factory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass(slots=True)
class BusterAnswer:
    user_input: str
    completion: Completion
    validator: Validator = None
    error: bool | None = None
    _documents_relevant: bool | None = None
    matched_documents: pd.DataFrame | None = None

    @property
    def documents_relevant(self):
        """Calls the validator to check if sources were used or not."""
        if self._documents_relevant is None:
            logger.info("checking for document relevance")
            self._documents_relevant = self.validator.check_sources_used(self.completion)
        return self._documents_relevant

    def to_json():
        pass


@dataclass
class BusterConfig:
    """Configuration object for a chatbot."""

    document_source: str = ""

    validator_cfg: dict = field(
        default_factory=lambda: {
            "unknown_prompt": "I Don't know how to answer your question.",
            "unknown_threshold": 0.85,
            "embedding_model": "text-embedding-ada-002",
        }
    )
    tokenizer_cfg: dict = field(
        default_factory=lambda: {
            "model_name": "gpt-3.5-turbo",
        }
    )
    retriever_cfg: dict = field(
        default_factory=lambda: {
            "max_tokens": 3000,
            "top_k": 3,
            "thresh": 0.7,
            "embedding_model": "text-embedding-ada-002",
        }
    )
    prompt_cfg: dict = field(
        default_factory=lambda: {
            "max_tokens": 3500,
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

    @lru_cache
    def get_embedding(self, query: str, engine: str):
        logger.info("generating embedding")
        return get_embedding(query, engine=engine)

    def update_cfg(self, cfg: BusterConfig):
        """Every time we set a new config, we update the things that need to be updated."""
        logger.info(f"Updating config to {cfg.document_source}:\n{cfg}")
        self._cfg = cfg
        self.document_source = cfg.document_source

        self.retriever_cfg = cfg.retriever_cfg
        self.completion_cfg = cfg.completion_cfg
        self.prompt_cfg = cfg.prompt_cfg
        self.tokenizer_cfg = cfg.tokenizer_cfg
        self.validator_cfg = cfg.validator_cfg

        # update all objects used
        self.validator = validator_factory(self.validator_cfg)
        self.tokenizer = tokenizer_factory(self.tokenizer_cfg)
        self.completer = completer_factory(self.completion_cfg)
        self.documents_formatter = document_formatter_factory(
            tokenizer=self.tokenizer,
            max_tokens=self.retriever_cfg["max_tokens"]
            # TODO: move max_tokens from retriever_cfg to somewhere more logical
        )
        self.prompt_formatter = prompt_formatter_factory(tokenizer=self.tokenizer, prompt_cfg=self.prompt_cfg)

        logger.info(f"Config Updated.")

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

    def process_input(self, user_input: str) -> BusterAnswer:
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
            engine=self.retriever_cfg["embedding_model"],
            source=self.document_source,
        )

        if len(matched_documents) == 0:
            logger.warning("No documents found...")

            def no_docs_completor():
                no_docs_msg = "No documents found."
                yield no_docs_msg

            completion = Completion(completor=no_docs_completor(), error=False)

            matched_documents = pd.DataFrame(columns=matched_documents.columns)
            answer = BusterAnswer(
                completion=completion,
                matched_documents=matched_documents,
                validator=self.validator,
                user_input=user_input,
            )
            return answer

        # format the matched documents, (will truncate them if too long)
        documents_str, matched_documents = self.documents_formatter.format(matched_documents)

        # prepare the prompt
        system_prompt = self.prompt_formatter.format(documents_str)

        completion = self.completer.generate_response(user_input=user_input, system_prompt=system_prompt)

        logger.info(f"GPT Response:\n{completion}")

        answer = BusterAnswer(
            completion=completion, matched_documents=matched_documents, validator=self.validator, user_input=user_input
        )
        return answer
