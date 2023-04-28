import logging
from dataclasses import dataclass

import pandas as pd

from buster.tokenizers import Tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class SystemPromptFormatter:
    tokenizer: Tokenizer
    max_tokens: 3500
    text_before_docs: str
    text_after_docs: str

    def format(
        self,
        documents: str,
    ) -> str:
        """
        Prepare the system prompt with prompt engineering.
        """
        system_prompt = self.text_before_docs + documents + self.text_after_docs

        if self.tokenizer.num_tokens(system_prompt) > self.max_tokens:
            raise ValueError(f"System prompt tokens > {self.max_tokens=}")
        return system_prompt


def prompt_formatter_factory(tokenizer: Tokenizer, prompt_cfg):
    return SystemPromptFormatter(
        tokenizer=tokenizer,
        max_tokens=prompt_cfg["max_tokens"],
        text_before_docs=prompt_cfg["text_before_documents"],
        text_after_docs=prompt_cfg["text_before_prompt"],
    )
