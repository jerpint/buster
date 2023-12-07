import logging
from dataclasses import dataclass

import pandas as pd

from buster.tokenizers import Tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class PromptFormatter:
    tokenizer: Tokenizer
    max_tokens: int
    text_before_docs: str
    text_after_docs: str
    formatter: str = "{text_before_docs}\n{documents}\n{text_after_docs}"

    def format(self, documents: str) -> str:
        """Formats the system prompt with prompt engineering.

        Joins the text before and after documents with the documents provided.

        Args:
            documents (str): The already formatted documents to include in the system prompt.

        Returns:
            str: The formatted system prompt.

        Raises:
            ValueError: If the number of prompt tokens exceeds the maximum allowed tokens.
        """
        system_prompt = self.formatter.format(
            text_before_docs=self.text_before_docs, documents=documents, text_after_docs=self.text_after_docs
        )

        if self.tokenizer.num_tokens(system_prompt) > self.max_tokens:
            raise ValueError(f"System prompt tokens > {self.max_tokens=}")
        return system_prompt


def prompt_formatter_factory(tokenizer: Tokenizer, prompt_cfg) -> PromptFormatter:
    """Creates a PromptFormatter instance.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for the PromptFormatter.
        prompt_cfg: The configuration for the PromptFormatter.

    Returns:
        PromptFormatter: The created PromptFormatter instance.
    """
    return PromptFormatter(
        tokenizer=tokenizer,
        max_tokens=prompt_cfg["max_tokens"],
        text_before_docs=prompt_cfg["text_before_documents"],
        text_after_docs=prompt_cfg["text_before_prompt"],
    )
