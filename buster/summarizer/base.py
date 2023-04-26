import copy
import logging

import openai

from .utils import word_count

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Summarizer:
    """Class useful for summarizing text."""

    def __init__(self, model, system_prompt, **completion_kwargs):
        self.model = model
        self.system_prompt = system_prompt
        self.completion_kwargs = completion_kwargs

    def summarize(self, text: str) -> str:
        assert isinstance(text, str)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]

        logger.info(f"Summarizing with the following prompt:\n{messages}")

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            **self.completion_kwargs,
        )
        summary = response["choices"][0]["message"]["content"]

        logger.info(f"SUMMARY:\n{summary}")
        return summary
