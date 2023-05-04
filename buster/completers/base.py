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
    completor: Iterator  # e.g. a response from openai.ChatCompletion
    error: bool
    text: str = ""
    error_msg: str | None = None


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

        self.completion = Completion(completor, self.error, text="")

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

    def complete(self, prompt, **completion_kwargs) -> str:
        try:
            response = openai.ChatCompletion.create(
                messages=prompt,
                **completion_kwargs,
            )

            self.error = False
            if completion_kwargs.get("stream") is True:

                def completor():
                    for chunk in response:
                        token: str = chunk["choices"][0]["delta"].get("content", "")
                        self.completion.text += token
                        yield token

            else:

                def completor():
                    full_response: str = response["choices"][0]["message"]["content"]
                    yield full_response

            return completor()

        except openai.error.InvalidRequestError:
            self.error = True
            error_msg = "Something went wrong with the request, try again soon! If the problem persists, contact the project admin."
            logger.exception("Invalid request to OpenAI API. See traceback:")

            def completor():
                yield error_msg

            return completor()

        except Exception as e:
            self.error = True
            error_msg = "Something went wrong with the request, try again soon! If the problem persists, contact the project admin."
            logger.exception("Unknown error when attempting to connect to OpenAI API. See traceback:")

            def completor():
                yield error_msg

            return completor()


def completer_factory(completer_cfg):
    name = completer_cfg["name"]
    completers = {
        "GPT3": GPT3Completer,
        "ChatGPT": ChatGPTCompleter,
    }
    return completers[name](completer_cfg["completion_kwargs"])
