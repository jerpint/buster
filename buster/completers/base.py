from abc import ABC, abstractmethod
import logging
import os

import openai
import promptlayer

from buster.formatter.base import Response

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


class Completer(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def complete(self, prompt) -> str:
        ...

    def generate_response(self, user_input, documents) -> Response:
        # Call the API to generate a response
        prompt = self.prepare_prompt(user_input, documents)
        name = self.cfg["name"]
        logger.info(f"querying model {name}...")
        logger.info(f"{prompt=}")
        try:
            completion_kwargs = self.cfg["completion_kwargs"]
            completion = self.complete(prompt=prompt, **completion_kwargs)
        except Exception as e:
            # log the error and return a generic response instead.
            logger.exception("Error connecting to OpenAI API. See traceback:")
            return Response("", True, "We're having trouble connecting to OpenAI right now... Try again soon!")

        return Response(completion)


class GPT3Completer(Completer):
    def prepare_prompt(
        self,
        user_input: str,
        documents: str,
    ) -> str:
        """
        Prepare the prompt with prompt engineering.
        """
        text_before_docs = self.cfg["text_before_documents"]
        text_before_prompt = self.cfg["text_before_prompt"]
        return text_before_docs + documents + text_before_prompt + user_input

    def complete(self, prompt, **completion_kwargs):
        response = openai.Completion.create(prompt=prompt, **completion_kwargs)
        return response["choices"][0]["text"]


class ChatGPTCompleter(Completer):
    def prepare_prompt(
        self,
        user_input: str,
        documents: str,
    ) -> list:
        """
        Prepare the prompt with prompt engineering.
        """
        text_before_docs = self.cfg["text_before_documents"]
        text_before_prompt = self.cfg["text_before_prompt"]
        prompt = [
            {"role": "system", "content": text_before_docs + documents + text_before_prompt},
            {"role": "user", "content": user_input},
        ]
        return prompt

    def complete(self, prompt, **completion_kwargs) -> str:
        response = openai.ChatCompletion.create(
            messages=prompt,
            **completion_kwargs,
        )

        return response["choices"][0]["message"]["content"]


def get_completer(completer_cfg):
    name = completer_cfg["name"]
    completers = {
        "GPT3": GPT3Completer,
        "ChatGPT": ChatGPTCompleter,
    }
    return completers[name](completer_cfg)
