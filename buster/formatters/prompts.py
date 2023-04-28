import logging
from dataclasses import dataclass

import pandas as pd

from buster.tokenizers import Tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class SystemPromptFormatter:
    tokenizer: Tokenizer
    text_before_docs: str
    text_after_docs: str
    max_tokens: int = 3000

    def format_documents(self, matched_documents: pd.DataFrame, max_tokens: int) -> str:
        # gather the documents in one large plaintext variable
        documents_list = matched_documents.content.to_list()
        documents_str = ""
        for idx, doc in enumerate(documents_list):
            documents_str += f"<DOCUMENT> {doc} <\\DOCUMENT>"

        token_count, encoded = self.tokenizer.num_tokens(documents_str, return_encoded=True)
        logger.info(f"token_count={token_count}")
        if token_count > max_tokens:
            logger.warning("truncating documents to fit...")
            documents_str = self.tokenizer.decode(encoded[0:max_tokens])
            logger.warning(f"Documents after truncation: {documents_str}")

        return documents_str

    def format(
        self,
        matched_documents: str,
    ) -> str:
        """
        Prepare the system prompt with prompt engineering.
        """
        documents = self.format_documents(matched_documents, max_tokens=self.max_tokens)
        system_prompt = self.text_before_docs + documents + self.text_after_docs
        return system_prompt


def prompt_formatter_factory(prompt_cfg, tokenizer: Tokenizer):
    return SystemPromptFormatter(
        tokenizer=tokenizer,
        text_before_docs=prompt_cfg["text_before_documents"],
        text_after_docs=prompt_cfg["text_before_prompt"],
        max_tokens=prompt_cfg["max_tokens"],
    )
