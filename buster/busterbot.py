import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from buster.completers import Completer, Completion, DocumentAnswerer
from buster.retriever import Retriever
from buster.validators import Validator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class BusterConfig:
    """Configuration object for a chatbot."""

    validator_cfg: dict = field(
        default_factory=lambda: {
            "unknown_prompts": [
                "I Don't know how to answer your question.",
            ],
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
    documents_formatter_cfg: dict = (
        field(
            default_factory=lambda: {
                "max_tokens": 3500,
                "formatter": "{content}",
            }
        ),
    )
    documents_answerer_cfg: dict = field(
        default_factory=lambda: {
            "no_documents_message": "No documents are available for this question.",
        }
    )
    completion_cfg: dict = field(
        default_factory=lambda: {
            "completion_kwargs": {
                "engine": "gpt-3.5-turbo",
                "temperature": 0,
                "stream": True,
            },
        }
    )


class Buster:
    def __init__(self, retriever: Retriever, document_answerer: DocumentAnswerer, validator: Validator):
        self.document_answerer = document_answerer
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

        # The returned message is either a generic invalid question message or an error handling message
        question_relevant, irrelevant_question_message = self.validator.check_question_relevance(user_input)

        if question_relevant:
            # question is relevant, get completor to generate completion
            matched_documents = self.retriever.retrieve(user_input, source=source)
            completion: Completion = self.document_answerer.get_completion(
                user_input=user_input,
                matched_documents=matched_documents,
                validator=self.validator,
                question_relevant=question_relevant,
            )

        else:
            # question was determined irrelevant, so we instead return a generic response set by the user.
            completion = Completion(
                error=False,
                user_input=user_input,
                matched_documents=pd.DataFrame(),
                answer_text=irrelevant_question_message,
                answer_relevant=False,
                question_relevant=False,
                validator=self.validator,
            )

        logger.info(f"Completion:\n{completion}")

        return completion
