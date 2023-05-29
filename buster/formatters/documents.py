import logging
from dataclasses import dataclass

import pandas as pd

from buster.tokenizers import Tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class DocumentsFormatter:
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

        return documents_str, matched_documents


def documents_formatter_factory(tokenizer: Tokenizer, max_tokens: int, formatter: str) -> DocumentsFormatter:
    return DocumentsFormatter(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        formatter=formatter,
    )
