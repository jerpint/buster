import logging
from dataclasses import dataclass

import pandas as pd

from buster.tokenizers import Tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class SystemPromptFormatter:
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
        return system_prompt


def prompt_formatter_factory(prompt_cfg):
    return SystemPromptFormatter(
        text_before_docs=prompt_cfg["text_before_documents"],
        text_after_docs=prompt_cfg["text_before_prompt"],
    )
