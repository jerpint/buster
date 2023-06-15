import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from buster.completers import Completer, Completion
from buster.retriever import Retriever
from buster.validators import Validator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class BusterConfig:
    """Configuration object for a chatbot."""

    validator_cfg: dict = field(
        default_factory=lambda: {
            "unknown_prompt": "I Don't know how to answer your question.",
            "unknown_threshold": 0.85,
            "embedding_model": "text-embedding-ada-002",
            "use_reranking": True,
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
    prompt_formatter_cfg: dict = field(
        default_factory=lambda: {
            "max_tokens": 3500,
            "text_before_docs": "You are a chatbot answering questions.\n",
            "text_after_docs": "Answer the following question:\n",
            "formatter": "{text_before_docs}\n{documents}\n{text_after_docs}",
        }
    )
    documents_formatter_cfg: dict = field(
        default_factory=lambda: {
            "max_tokens": 3500,
            "formatter": "{content}",
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
    def __init__(self, retriever: Retriever, completer: Completer, validator: Validator):
        self.completer = completer
        self.retriever = retriever
        self.validator = validator

    def process_input(self, user_input: str, source: str = None) -> Completion:
        """
        Main function to process the input question and generate a formatted output.
        """

        logger.info(f"User Input:\n{user_input}")


        # We make sure there is always a newline at the end of the question to avoid completing the question.
        if not user_input.endswith("\n"):
            user_input += "\n"

        question_relevant = self.validator.check_question_relevance(user_input)

        if question_relevant:
            matched_documents = self.retriever.retrieve(user_input, source=source)

            completion = self.completer.get_completion(user_input=user_input, matched_documents=matched_documents)

        else:
            completion = Completion(
                error=False,
                user_input=user_input,
                matched_documents=pd.DataFrame(),
                completor = self.validator.invalid_question_response,
                answer_relevant = False,
            )

        completion.question_relevant = question_relevant
        logger.info(f"Completion:\n{completion}")

        return completion

    def postprocess_completion(self, completion) -> Completion:
        """This will check if the answer is relevant, and rerank the sources by relevance too."""
        return self.validator.validate(completion=completion)
