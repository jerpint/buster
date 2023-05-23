import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

import openai
import pandas as pd
import promptlayer
from fastapi.encoders import jsonable_encoder

from buster.formatters.documents import DocumentsFormatter
from buster.formatters.prompts import PromptFormatter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Check if an API key exists for promptlayer, if it does, use it
promptlayer_api_key = os.environ.get("PROMPTLAYER_API_KEY")
if promptlayer_api_key:
    logger.info("Enabling prompt layer...")
    promptlayer.api_key = promptlayer_api_key

    # replace openai with the promptlayer wrapper
    openai = promptlayer.openai
    openai.api_key = os.environ.get("OPENAI_API_KEY")


@dataclass
class Completion:
    error: bool
    user_input: str
    matched_documents: pd.DataFrame
    completor: Iterator | str
    answer_relevant: bool = None

    # private property, should not be set at init
    _completor: Iterator | str = field(init=False, repr=False)  # e.g. a streamed response from openai.ChatCompletion
    _text: str = None

    @property
    def text(self):
        if self._text is None:
            # generates the text if it wasn't already generated
            self._text = "".join([i for i in self.completor])
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        self._text = value

    @property
    def completor(self):
        if isinstance(self._completor, str):
            # convert str to iterator
            self._completor = (msg for msg in self._completor)

        # keeps track of the yielded text
        self._text = ""
        for token in self._completor:
            self._text += token
            yield token

    @completor.setter
    def completor(self, value: str) -> None:
        self._completor = value

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
            "text": self.text,
            "matched_documents": self.matched_documents,
            "answer_relevant": self.answer_relevant,
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
    def __init__(
        self,
        documents_formatter: DocumentsFormatter,
        prompt_formatter: PromptFormatter,
        completion_kwargs: dict,
        no_documents_message: str = "No documents were found that match your question.",
    ):
        self.completion_kwargs = completion_kwargs
        self.documents_formatter = documents_formatter
        self.prompt_formatter = prompt_formatter
        self.no_documents_message = no_documents_message

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

    def generate_response(self, user_input: str, matched_documents: pd.DataFrame):
        # Call the API to generate a response

        logger.info(f"{user_input=}")

        if len(matched_documents) == 0:
            logger.warning("no documents found...")
            # no document was found, pass the appropriate message instead...

            # empty dataframe
            matched_documents = pd.DataFrame(columns=matched_documents.columns)

            completion = Completion(
                user_input=user_input,
                completor=self.no_documents_message,
                error=False,
                matched_documents=matched_documents,
            )
            return completion

        # prepare the prompt
        prompt = self.prepare_prompt(matched_documents)
        logger.info(f"{prompt=}")

        logger.info(f"querying model with parameters: {self.completion_kwargs}...")
        completor = self.complete(prompt=prompt, user_input=user_input, **self.completion_kwargs)

        completion = Completion(
            completor=completor, error=self.error, matched_documents=matched_documents, user_input=user_input
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

                def completor():
                    for chunk in response:
                        token: str = chunk["choices"][0].get("text")
                        yield token

                return completor()
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
        try:
            response = openai.ChatCompletion.create(
                messages=messages,
                **completion_kwargs,
            )

            self.error = False
            if completion_kwargs.get("stream") is True:
                # We are entering streaming mode, so here were just wrapping the streamed
                # openai response to be easier to handle later
                def completor():
                    for chunk in response:
                        token: str = chunk["choices"][0]["delta"].get("content", "")
                        yield token

                return completor()

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
