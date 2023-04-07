from dataclasses import dataclass
import logging

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class SystemPromptFormatter:
    text_before_docs: str = ""
    text_after_docs: str = ""
    max_words: int = 4000

    def format_documents(self, matched_documents: pd.DataFrame, max_words: int) -> str:
        # gather the documents in one large plaintext variable
        documents_list = matched_documents.content.to_list()
        documents_str = ""
        for idx, doc in enumerate(documents_list):
            documents_str += f"<DOCUMENT> {doc} <\DOCUMENT>"

        # truncate the documents to fit
        # TODO: increase to actual token count
        word_count = len(documents_str.split(" "))
        if word_count > max_words:
            logger.warning("truncating documents to fit...")
            documents_str = " ".join(documents_str.split(" ")[0:max_words])
            logger.warning(f"Documents after truncation: {documents_str}")

        return documents_str

    def format(
        self,
        matched_documents: str,
    ) -> str:
        """
        Prepare the system prompt with prompt engineering.
        """
        documents = self.format_documents(matched_documents, max_words=self.max_words)
        system_prompt = self.text_before_docs + documents + self.text_after_docs
        return system_prompt