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


@dataclass
class Completion:
    error: bool
    user_input: str
    matched_documents: pd.DataFrame
    completor: Iterator | str
    answer_relevant: bool = None
    question_relevant: bool = None

    # private property, should not be set at init
    _completor: Iterator | str = field(init=False, repr=False)  # e.g. a streamed response from openai.ChatCompletion
    _text: str = (
        None  # once the generator of the completor is exhausted, the text will be available in the self.text property
    )

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
            "text": self.text,
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
        completion_dict["completor"] = completion_dict["text"]
        del completion_dict["text"]

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

    def get_completion(self, user_input: str, matched_documents: pd.DataFrame) -> Completion:
        """Generate a completion to a user's question based on matched documents."""

        # The completor assumes a question was previously determined valid, otherwise it would not be called.
        question_relevant = True

        logger.info(f"{user_input=}")

        if len(matched_documents) == 0:
            # no document was found, pass the appropriate message instead...
            logger.warning("no documents found...")

            # empty dataframe
            matched_documents = pd.DataFrame(columns=matched_documents.columns)

            # because we are proceeding with a completion, we assume the question is relevant.
            completion = self.completion_class(
                user_input=user_input,
                completor=self.no_documents_message,
                error=False,
                matched_documents=matched_documents,
                question_relevant=question_relevant,
            )
            return completion

        # prepare the prompt with matched documents
        prompt = self.prepare_prompt(matched_documents)
        logger.info(f"{prompt=}")

        logger.info(f"querying model with parameters: {self.completion_kwargs}...")
        completor = self.complete(prompt=prompt, user_input=user_input, **self.completion_kwargs)

        completion = self.completion_class(
            completor=completor,
            error=self.error,
            matched_documents=matched_documents,
            user_input=user_input,
            question_relevant=question_relevant,
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
        self.error = False
        try:
            response = openai.ChatCompletion.create(
                messages=messages,
                **completion_kwargs,
            )

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
