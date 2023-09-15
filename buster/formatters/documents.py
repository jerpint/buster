import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from buster.tokenizers import Tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DocumentsFormatter(ABC):
    """
    Abstract base class for document formatters.

    Subclasses are required to implement the `format` method which transforms the input documents
    into the desired format.
    """

    @abstractmethod
    def format(self, matched_documents: pd.DataFrame) -> tuple[str, pd.DataFrame]:
        """
        Abstract method to format matched documents.

        Parameters:
        - matched_documents (pd.DataFrame): DataFrame containing the matched documents to be formatted.

        Returns:
        - tuple[str, pd.DataFrame]: A tuple containing the formatted documents as a string and
                                    the possibly truncated matched documents DataFrame.
        """
        pass


@dataclass
class DocumentsFormatterHTML(DocumentsFormatter):
    """
    Formatter class to convert matched documents into an HTML format.

    Attributes:
    - tokenizer (Tokenizer): Tokenizer instance to count tokens in the documents.
    - max_tokens (int): Maximum allowed tokens for the formatted documents.
    - formatter (str): String formatter for the document's content.
    - inner_tag (str): HTML tag that will be used at the document level.
    - outer_tag (str): HTML tag that will be used at the documents collection level.
    """

    tokenizer: Tokenizer
    max_tokens: int
    formatter: str = "{content}"
    inner_tag: str = "DOCUMENT"
    outer_tag: str = "DOCUMENTS"

    def format(self, matched_documents: pd.DataFrame) -> tuple[str, pd.DataFrame]:
        """
        Format the matched documents into an HTML format.

        If the total tokens exceed max_tokens, documents are truncated or omitted to fit within the limit.

        Parameters:
        - matched_documents (pd.DataFrame): DataFrame containing the matched documents to be formatted.

        Returns:
        - tuple[str, pd.DataFrame]: A tuple containing the formatted documents as an HTML string and
                                    the possibly truncated matched documents DataFrame.
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
                documents_str += f"<{self.inner_tag}>{doc}<\\{self.inner_tag}>"
                total_tokens += token_count
            else:
                logger.warning("truncating document to fit...")
                remaining_tokens = max_tokens - total_tokens
                truncated_doc = self.tokenizer.decode(encoded[:remaining_tokens])
                documents_str += f"<{self.inner_tag}>{truncated_doc}<\\{self.inner_tag}>"
                logger.warning(f"Documents after truncation: {documents_str}")
                break

        if num_preserved_docs < (num_total_docs):
            logger.warning(
                f"{num_preserved_docs}/{num_total_docs} documents were preserved from the matched documents due to truncation."
            )
            matched_documents = matched_documents.iloc[:num_preserved_docs]

        documents_str = f"<{self.outer_tag}>{documents_str}<\\{self.outer_tag}>"

        return documents_str, matched_documents


@dataclass
class DocumentsFormatterJSON(DocumentsFormatter):
    """
    Formatter class to convert matched documents into a JSON format.

    Attributes:
    - tokenizer (Tokenizer): Tokenizer instance to count tokens in the documents.
    - max_tokens (int): Maximum allowed tokens for the formatted documents.
    - columns (list[str]): List of columns to include in the JSON format.
    """

    tokenizer: Tokenizer
    max_tokens: int
    columns: list[str]

    def format(self, matched_documents: pd.DataFrame) -> tuple[str, pd.DataFrame]:
        """
        Format the matched documents into a JSON format.

        If the total tokens exceed max_tokens, documents are omitted one at a time until it fits the limit.

        Parameters:
        - matched_documents (pd.DataFrame): DataFrame containing the matched documents to be formatted.

        Returns:
        - tuple[str, pd.DataFrame]: A tuple containing the formatted documents as a JSON string and
                                    the possibly truncated matched documents DataFrame.
        """

        max_tokens = self.max_tokens
        documents_str = matched_documents[self.columns].to_json(orient="records")
        token_count, _ = self.tokenizer.num_tokens(documents_str, return_encoded=True)

        while token_count > max_tokens:
            # Truncated too much, no documents left, raise an error
            if len(matched_documents) == 0:
                raise ValueError(
                    f"Could not truncate documents to fit {max_tokens=}. Consider increasing max_tokens or decreasing chunk lengths."
                )

            # Too many tokens, drop a document and try again.
            matched_documents = matched_documents.iloc[:-1]
            documents_str = matched_documents[self.columns].to_json(orient="records")
            token_count, _ = self.tokenizer.num_tokens(documents_str, return_encoded=True)

            # Log a warning with more details
            logger.warning(
                f"Truncating documents to fit. Remaining documents after truncation: {len(matched_documents)}"
            )

        return documents_str, matched_documents
