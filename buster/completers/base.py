import logging
import os
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
        answer_generator: Iterator | str,
        validator=None,
        answer_relevant: bool = None,
        question_relevant: bool = None,
    ):
        self.error = error
        self.user_input = user_input
        self.matched_documents = matched_documents
        self.validator = validator
        self._answer_generator = answer_generator
        self._answer_relevant = answer_relevant
        self._question_relevant = question_relevant
        self._answer_text = None

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
    def answer_generator(self, message: str | Iterator) -> None:
        if isinstance(message, str):
            # convert str to iterator
            self._answer_generator = (msg for msg in message)
        self._answer_generator = message

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

        # avoids setting a property at init. the .text method will still be available.
        completion_dict["answer_generator"] = completion_dict["answer_text"]
        del completion_dict["answer_text"]

        return cls(**completion_dict)


class Completer(ABC):
    def __init__(
        self,
        documents_formatter: DocumentsFormatter,
        prompt_formatter: PromptFormatter,
        completion_kwargs: dict,
        no_documents_message: str = "No documents were found that match your question.",
        completion_class: Completion = Completion,
    ):
        self.completion_kwargs = completion_kwargs
        self.documents_formatter = documents_formatter
        self.prompt_formatter = prompt_formatter
        self.no_documents_message = no_documents_message
        self.completion_class = completion_class

    @abstractmethod
    def complete(self, prompt: str, user_input: str) -> Completion:
        ...

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
            # no document was found, pass the appropriate message instead...
            logger.warning("no documents found...")

            # empty dataframe
            matched_documents = pd.DataFrame(columns=matched_documents.columns)

            # because we are proceeding with a completion, we assume the question is relevant.
            completion = self.completion_class(
                user_input=user_input,
                answer_generator=self.no_documents_message,
                error=False,
                matched_documents=matched_documents,
                question_relevant=question_relevant,
                validator=validator,
            )
            return completion

        # prepare the prompt with matched documents
        prompt = self.prepare_prompt(matched_documents)
        logger.info(f"{prompt=}")

        logger.info(f"querying model with parameters: {self.completion_kwargs}...")
        answer_generator = self.complete(prompt=prompt, user_input=user_input, **self.completion_kwargs)

        completion = self.completion_class(
            answer_generator=answer_generator,
            error=self.error,
            matched_documents=matched_documents,
            user_input=user_input,
            question_relevant=question_relevant,
            validator=validator,
        )

        return completion


class GPT3Completer(Completer):
    # TODO: Adapt...
    def prepare_prompt(
        self,
        system_prompt: str,
        user_input: str,
    ) -> str:
        """
        Prepare the prompt with prompt engineering.
        """
        return system_prompt + user_input

    def complete(self, prompt, user_input, **completion_kwargs):
        prompt = prompt + user_input
        try:
            response = openai.Completion.create(prompt=prompt, **completion_kwargs)
            self.error = False
            if completion_kwargs.get("stream") is True:

                def answer_generator():
                    for chunk in response:
                        token: str = chunk["choices"][0].get("text")
                        yield token

                return answer_generator()
            else:
                return response["choices"][0]["text"]
        except Exception as e:
            logger.exception(e)
            self.error = True
            error_msg = "Something went wrong..."
            return error_msg


class ChatGPTCompleter(Completer):
    def complete(self, prompt: str, user_input, **completion_kwargs) -> str | Iterator:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ]
        self.error = False
        try:
            response = openai.ChatCompletion.create(
                messages=messages,
                **completion_kwargs,
            )

            if completion_kwargs.get("stream") is True:
                # We are entering streaming mode, so here were just wrapping the streamed
                # openai response to be easier to handle later
                def answer_generator():
                    for chunk in response:
                        token: str = chunk["choices"][0]["delta"].get("content", "")
                        yield token

                return answer_generator()

            else:
                full_response: str = response["choices"][0]["message"]["content"]
                return full_response

        except openai.error.InvalidRequestError:
            self.error = True
            logger.exception("Invalid request to OpenAI API. See traceback:")
            error_msg = "Something went wrong with the request, try again soon! If the problem persists, contact the project admin."
            return error_msg

        except openai.error.RateLimitError:
            self.error = True
            logger.exception("RateLimit error from OpenAI. See traceback:")
            error_msg = "OpenAI servers seem to be overloaded, try again later!"
            return error_msg

        except Exception as e:
            self.error = True
            error_msg = "Something went wrong with the request, try again soon! If the problem persists, contact the project admin."
            logger.exception("Unknown error when attempting to connect to OpenAI API. See traceback:")
            return error_msg


def completer_factory(completer_cfg):
    name = completer_cfg["name"]
    completers = {
        "GPT3": GPT3Completer,
        "ChatGPT": ChatGPTCompleter,
    }
    return completers[name](completer_cfg["completion_kwargs"])
