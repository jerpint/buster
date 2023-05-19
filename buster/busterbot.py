import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from fastapi.encoders import jsonable_encoder

from buster.completers.base import Completer, Completion
from buster.completers import completer_factory
from buster.formatters.documents import documents_formatter_factory
from buster.formatters.prompts import prompt_formatter_factory
from buster.retriever import Retriever

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class BusterAnswer:
    pass


@dataclass
class BusterAnswerData:
    user_input: str
    completion: Completion
    matched_documents: pd.DataFrame
    response_relevant: bool | None = None

    @classmethod
    def from_dict(cls, answer_dict: dict):
        if isinstance(answer_dict["matched_documents"], str):
            answer_dict["matched_documents"] = pd.read_json(answer_dict["matched_documents"], orient="index")
        elif isinstance(answer_dict["matched_documents"], dict):
            answer_dict["matched_documents"] = pd.DataFrame(answer_dict["matched_documents"]).T
        else:
            raise ValueError(f"Unknown type for matched_documents: {type(answer_dict['matched_documents'])}")
        answer_dict["completion"] = Completion.from_dict(answer_dict["completion"])
        return cls(**answer_dict)

    def to_json(self) -> Any:
        def encode_df(df: pd.DataFrame) -> dict:
            if "embedding" in df.columns:
                df = df.drop(columns=["embedding"])
            return df.to_json(orient="index")

        custom_encoder = {
            # Converts the matched_documents in the user_responses to json
            pd.DataFrame: encode_df,
        }

        to_encode = {
            "user_input": self.user_input,
            "completion": self.completion.to_json(),
            "matched_documents": self.matched_documents,
            "response_relevant": self.response_relevant,
        }
        return jsonable_encoder(to_encode, custom_encoder=custom_encoder)


@dataclass
class BusterConfig:
    """Configuration object for a chatbot."""

    document_source: str = ""

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
    prompt_cfg: dict = field(
        default_factory=lambda: {
            "max_tokens": 3500,
            "text_before_documents": "You are a chatbot answering questions.\n",
            "text_before_prompt": "Answer the following question:\n",
        }
    )
    documents_formatter_cfg: dict = field(
        default_factory=lambda: {
            "max_tokens": 3500,
            "format_str": "{content}",
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
    def __init__(self, retriever: Retriever, completer: Completer, validator):
        self.completer = completer
        self.retriever = retriever
        self.validator = validator


    def process_input(self, user_input: str, source: str) -> BusterAnswer:
        """
        Main function to process the input question and generate a formatted output.
        """

        logger.info(f"User Input:\n{user_input}")

        # We make sure there is always a newline at the end of the question to avoid completing the question.
        if not user_input.endswith("\n"):
            user_input += "\n"

        matched_documents = self.retriever.retrieve(user_input, source=source)

        completion = self.completer.generate_response(user_input=user_input, matched_documents=matched_documents)

        logger.info(f"Completion:\n{completion}")

        return completion

    def postprocess_completion(self, completion):
        return self.validator.validate(completion=completion)
