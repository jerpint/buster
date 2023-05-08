import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import openai
import promptlayer

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


@dataclass(slots=True)
class Completion:
    _completor: Iterator | str  # e.g. a streamed response from openai.ChatCompletion
    error: bool
    _text: str | None = None

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


class Completer(ABC):
    def __init__(self, completion_kwargs: dict):
        self.completion_kwargs = completion_kwargs

    @abstractmethod
    def complete(self, prompt) -> str:
        ...

    def generate_response(self, system_prompt, user_input):
        # Call the API to generate a response
        prompt = self.prepare_prompt(system_prompt, user_input)
        logger.info(f"querying model with parameters: {self.completion_kwargs}...")
        logger.info(f"{system_prompt=}")
        logger.info(f"{user_input=}")

        completor = self.complete(prompt=prompt, **self.completion_kwargs)

        self.completion = Completion(completor, self.error)

        return self.completion


class GPT3Completer(Completer):
    # TODO: Update
    def prepare_prompt(
        self,
        system_prompt: str,
        user_input: str,
    ) -> str:
        """
        Prepare the prompt with prompt engineering.
        """
        return system_prompt + user_input

    def complete(self, prompt, **completion_kwargs):
        response = openai.Completion.create(prompt=prompt, **completion_kwargs)
        return response["choices"][0]["text"]


class ChatGPTCompleter(Completer):
    def prepare_prompt(
        self,
        system_prompt: str,
        user_input: str,
    ) -> list:
        """
        Prepare the prompt with prompt engineering.
        """
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        return prompt

    def complete(self, prompt, **completion_kwargs) -> str | Iterator:
        try:
            response = openai.ChatCompletion.create(
                messages=prompt,
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
