from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass

import pandas as pd

from buster.tokenizers import Tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DocumentsFormatter(ABC):
    @abstractmethod
    def format():
        ...


@dataclass
class DocumentsFormatterHTML(DocumentsFormatter):
    tokenizer: Tokenizer
    max_tokens: int
    formatter: str = "{content}"

    def format(
        self,
        matched_documents: pd.DataFrame,
    ) -> tuple[str, pd.DataFrame]:
        """Format our matched documents to plaintext.

        We also make sure they fit in the alloted max_tokens space.
        """
        documents_str = ""
        total_tokens = 0
        max_tokens = self.max_tokens

        num_total_docs = len(matched_documents)
        num_preserved_docs = 0
        for _, row in matched_documents.iterrows():
            doc = self.formatter.format_map(row.to_dict())
            num_preserved_docs += 1
            token_count, encoded = self.tokenizer.num_tokens(doc, return_encoded=True)
            if total_tokens + token_count <= max_tokens:
                documents_str += f"<DOCUMENT>{doc}<\\DOCUMENT>"
                total_tokens += token_count
            else:
                logger.warning("truncating document to fit...")
                remaining_tokens = max_tokens - total_tokens
                truncated_doc = self.tokenizer.decode(encoded[:remaining_tokens])
                documents_str += f"<DOCUMENT>{truncated_doc}<\\DOCUMENT>"
                logger.warning(f"Documents after truncation: {documents_str}")
                break

        if num_preserved_docs < (num_total_docs):
            logger.warning(
                f"{num_preserved_docs}/{num_total_docs} documents were preserved from the matched documents due to truncation."
            )
            matched_documents = matched_documents.iloc[:num_preserved_docs]

        documents_str = f"<DOCUMENTS>{documents_str}<\\DOCUMENTS>"

        return documents_str, matched_documents


@dataclass
class DocumentsFormatterJSON(DocumentsFormatter):
    tokenizer: Tokenizer
    max_tokens: int
    columns: list[str]

    def format(
        self,
        matched_documents: pd.DataFrame,
    ) -> tuple[str, pd.DataFrame]:
        """Format our matched documents to plaintext.

        We also make sure they fit in the alloted max_tokens space.
        """
        documents_str = ""
        max_tokens = self.max_tokens

        documents_str = matched_documents[self.columns].to_json(orient="records")
        token_count, _ = self.tokenizer.num_tokens(documents_str, return_encoded=True)

        while token_count > max_tokens:
            # Too many tokens, drop a document and try again.
            logger.warning("truncating document to fit...")
            matched_documents = matched_documents.iloc[:-1]
            documents_str = matched_documents[["content", "source"]].to_json(orient="records")
            token_count, _ = self.tokenizer.num_tokens(documents_str, return_encoded=True)
            logger.warning(f"Documents after truncation: {documents_str}")

        return documents_str, matched_documents
