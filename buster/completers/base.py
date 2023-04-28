import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    text: str
    error: bool = False
    error_msg: str | None = None


class Completer(ABC):
    def __init__(self, completion_kwargs: dict):
        self.completion_kwargs = completion_kwargs

    @abstractmethod
    def complete(self, prompt) -> str:
        ...

    def generate_response(self, system_prompt, user_input) -> Completion:
        # Call the API to generate a response
        prompt = self.prepare_prompt(system_prompt, user_input)
        logger.info(f"querying model with parameters: {self.completion_kwargs}...")
        logger.info(f"{system_prompt=}")
        logger.info(f"{user_input=}")
        try:
            completion = self.complete(prompt=prompt, **self.completion_kwargs)
        except openai.error.InvalidRequestError:
            logger.exception("Invalid request to OpenAI API. See traceback:")
            return Completion("Something went wrong, try again soon!", True, "Invalid request made to openai.")
        except Exception as e:
            # log the error and return a generic response instead.
            logger.exception("Error connecting to OpenAI API. See traceback:")
            return Completion(
                "Something went wrong, try again soon!", True, "Unexpected error at the generate_response level"
            )

        return Completion(completion)


class GPT3Completer(Completer):
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
        response = openai.ChatCompletion.create(
            messages=prompt,
            **completion_kwargs,
        )

        return response["choices"][0]["message"]["content"]


def completer_factory(completer_cfg):
    name = completer_cfg["name"]
    completers = {
        "GPT3": GPT3Completer,
        "ChatGPT": ChatGPTCompleter,
    }
    return completers[name](completer_cfg["completion_kwargs"])
