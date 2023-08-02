import logging
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import openai
import pandas as pd
from fastapi.encoders import jsonable_encoder

from buster.formatters.documents import DocumentsFormatter
from buster.formatters.prompts import PromptFormatter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Check if an API key exists for promptlayer, if it does, use it
promptlayer_api_key = os.environ.get("PROMPTLAYER_API_KEY")
if promptlayer_api_key:
    try:
        import promptlayer

        logger.info("Enabling prompt layer...")
        promptlayer.api_key = promptlayer_api_key

        # replace openai with the promptlayer wrapper
        openai = promptlayer.openai
    except Exception as e:
        logger.exception("Something went wrong enabling promptlayer.")

# Set openai credentials
openai.api_key = os.environ.get("OPENAI_API_KEY")


class Completion:
    def __init__(
        self,
        error: bool,
        user_input: str,
        matched_documents: pd.DataFrame,
        answer_generator: Optional[Iterator] = None,
        answer_text: Optional[str] = None,
        answer_relevant: Optional[bool] = None,
        question_relevant: Optional[bool] = None,
        completion_kwargs: Optional[dict] = None,
        validator=None,
    ):
        self.error = error
        self.user_input = user_input
        self.matched_documents = matched_documents
        self.validator = validator
        self.completion_kwargs = completion_kwargs
        self._answer_relevant = answer_relevant
        self._question_relevant = question_relevant

        self._validate_arguments(answer_generator, answer_text)

    def __repr__(self):
        class_name = type(self).__name__
        return (
            f"{class_name}("
            f"user_input={self.user_input!r}, "
            f"error={self.error!r}, "
            f"matched_documents={self.matched_documents!r}, "
            f"answer_text={self._answer_text!r}, "
            f"answer_generator={self.answer_generator!r}, "
            f"answer_relevant={self._answer_relevant!r}, "
            f"question_relevant={self.question_relevant!r}, "
            f"completion_kwargs={self.completion_kwargs!r}, "
            "),"
        )

    def _validate_arguments(self, answer_generator: Optional[Iterator], answer_text: Optional[str]):
        """Sets answer_generator and answer_text properties depending on the provided inputs.

        Checks that one of either answer_generator or answer_text is not None.
        If answer_text is set, a generator can simply be inferred from answer_text.
        If answer_generator is set, answer_text will be set only once the generator gets called. Set to None for now.
        """
        if (answer_generator is None and answer_text is None) or (
            answer_generator is not None and answer_text is not None
        ):
            raise ValueError("Only one of 'answer_generator' and 'answer_text' must be set.")

        # If text is provided, the genrator can be inferred
        if answer_text is not None:
            assert isinstance(answer_text, str)
            answer_generator = (msg for msg in answer_text)

        self._answer_text = answer_text
        self._answer_generator = answer_generator

    @property
    def answer_relevant(self) -> bool:
        """Property determining the relevance of an answer (bool).

        If an error occured, the relevance is False.
        If no documents were retrieved, the relevance is also False.
        Otherwise, the relevance is computed as defined by the validator (e.g. comparing to embeddings)
        """
        if self.error:
            self._answer_relevant = False
        elif len(self.matched_documents) == 0:
            self._answer_relevant = False
        elif self._answer_relevant is not None:
            return self._answer_relevant
        else:
            # Check the answer relevance by looking at the embeddings
            self._answer_relevant = self.validator.check_answer_relevance(self.answer_text)
        return self._answer_relevant

    @property
    def question_relevant(self):
        """Property determining the relevance of the question asked (bool)."""
        return self._question_relevant

    @property
    def answer_text(self):
        if self._answer_text is None:
            # generates the text if it wasn't already generated
            self._answer_text = "".join([i for i in self.answer_generator])
        return self._answer_text

    @answer_text.setter
    def answer_text(self, value: str) -> None:
        self._answer_text = value

    @property
    def answer_generator(self):
        # keeps track of the yielded text
        self._answer_text = ""
        for token in self._answer_generator:
            self._answer_text += token
            yield token

        self.postprocess()

    @answer_generator.setter
    def answer_generator(self, generator: Iterator) -> None:
        self._answer_generator = generator

    def postprocess(self):
        """Function executed after the answer text is generated by the answer_generator"""

        if self.validator is None:
            # TODO: This should only happen if declaring a Completion using .from_dict() method.
            # This behaviour is not ideal and we may want to remove support for .from_dict() in the future.
            logger.info("No validator was set, skipping postprocessing.")
            return

        if self.validator.use_reranking:
            # rerank docs in order of cosine similarity to the question
            self.matched_documents = self.validator.rerank_docs(
                answer=self.answer_text, matched_documents=self.matched_documents
            )

        # access the property so it gets set if not computed alerady
        self.answer_relevant

    def to_json(self, columns_to_ignore: Optional[list[str]] = None) -> Any:
        """Converts selected attributes of the object to a JSON format.

        Args:
            columns_to_ignore (list[str]): A list of column names to ignore in the csulting matched_documents dataframe.

        Returns:
            Any: The object's attributes encoded as JSON.

        Notes:
            - The 'matched_documents' attribute of type pd.DataFrame is encoded separately
            using a custom encoder.
            - The resulting JSON may exclude specified columns based on the 'columns_to_ignore' parameter.
        """

        def encode_df(df: pd.DataFrame) -> dict:
            if columns_to_ignore is not None:
                df = df.drop(columns=columns_to_ignore, errors="ignore")
            return df.to_json(orient="index")

        custom_encoder = {
            # Converts the matched_documents in the user_responses to json
            pd.DataFrame: encode_df,
        }

        to_encode = {
            "user_input": self.user_input,
            "answer_text": self.answer_text,
            "matched_documents": self.matched_documents,
            "answer_relevant": self.answer_relevant,
            "question_relevant": self.question_relevant,
            "completion_kwargs": self.completion_kwargs,
            "error": self.error,
        }
        return jsonable_encoder(to_encode, custom_encoder=custom_encoder)

    @classmethod
    def from_dict(cls, completion_dict: dict):
        if isinstance(completion_dict["matched_documents"], str):
            completion_dict["matched_documents"] = pd.read_json(completion_dict["matched_documents"], orient="index")
        elif isinstance(completion_dict["matched_documents"], dict):
            completion_dict["matched_documents"] = pd.DataFrame(completion_dict["matched_documents"]).T
        else:
            raise ValueError(f"Unknown type for matched_documents: {type(completion_dict['matched_documents'])}")

        return cls(**completion_dict)


class Completer(ABC):
    """Generic LLM-based completer. Requires a prompt and an input to produce an output."""

    def __init__(
        self,
        completion_kwargs: dict,
    ):
        self.completion_kwargs = completion_kwargs

    @abstractmethod
    def complete(self, prompt: str, user_input: str) -> Completion:
        ...


class DocumentAnswerer:
    """Completer that will answer questions based on documents.

    It takes care of formatting the prompts and the documents, and generating the answer when relevant.
    """

    def __init__(
        self,
        documents_formatter: DocumentsFormatter,
        prompt_formatter: PromptFormatter,
        completer: Completer,
        completion_class: Completion = Completion,
        no_documents_message: str = "No documents were found that match your question.",
    ):
        self.completer = completer
        self.documents_formatter = documents_formatter
        self.prompt_formatter = prompt_formatter
        self.no_documents_message = no_documents_message
        self.completion_class = completion_class

    def prepare_prompt(self, matched_documents) -> str:
        """Prepare the prompt with prompt engineering.

        A user's question is not included here. We use the documents formatter and prompt formatter to
        compose the prompt itself.
        """

        # format the matched documents, (will truncate them if too long)
        formatted_documents, _ = self.documents_formatter.format(matched_documents)
        prompt = self.prompt_formatter.format(formatted_documents)
        return prompt

    def get_completion(
        self, user_input: str, matched_documents: pd.DataFrame, validator, question_relevant: bool = True
    ) -> Completion:
        """Generate a completion to a user's question based on matched documents.

        It is safe to assume the question_relevance to be True if we made it here."""

        logger.info(f"{user_input=}")

        if len(matched_documents) == 0:
            warning_msg = "No documents found during retrieval."
            warnings.warn(warning_msg)
            logger.warning(warning_msg)

            # empty dataframe
            matched_documents = pd.DataFrame(columns=matched_documents.columns)

            # because we are requesting a completion, we assume the question is relevant.
            # However, no documents were found, so we pass the no documents found message instead of generating the answer.
            # The completion does not get triggered, so we do not pass completion kwargs here either.
            completion = self.completion_class(
                user_input=user_input,
                answer_text=self.no_documents_message,
                error=False,
                matched_documents=matched_documents,
                question_relevant=question_relevant,
                validator=validator,
            )
            return completion

        # prepare the prompt with matched documents
        prompt = self.prepare_prompt(matched_documents)
        logger.info(f"{prompt=}")

        logger.info(f"querying model with parameters: {self.completer.completion_kwargs}...")

        try:
            answer_generator = self.completer.complete(prompt=prompt, user_input=user_input)
            error = False

        except openai.error.InvalidRequestError:
            error = True
            logger.exception("Invalid request to OpenAI API. See traceback:")
            error_msg = "Something went wrong with the request, try again soon! If the problem persists, contact the project admin."
            return error_msg

        except openai.error.RateLimitError:
            error = True
            logger.exception("RateLimit error from OpenAI. See traceback:")
            error_msg = "OpenAI servers seem to be overloaded, try again later!"
            return error_msg

        except Exception as e:
            error = True
            error_msg = "Something went wrong with the request, try again soon! If the problem persists, contact the project admin."
            logger.exception("Unknown error when attempting to connect to OpenAI API. See traceback:")
            return error_msg

        completion = self.completion_class(
            answer_generator=answer_generator,
            error=error,
            matched_documents=matched_documents,
            user_input=user_input,
            question_relevant=question_relevant,
            validator=validator,
            completion_kwargs=self.completer.completion_kwargs,
        )

        return completion
