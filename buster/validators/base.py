import logging

import pandas as pd

from buster.llm_utils import cosine_similarity, get_openai_embedding
from buster.validators.validators import (
    AnswerValidator,
    DocumentsValidator,
    QuestionValidator,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Validator:
    def __init__(
        self,
        use_reranking: bool,
        validate_documents: bool,
        question_validator_cfg=None,
        answer_validator_cfg=None,
        documents_validator_cfg=None,
    ):
        """
        Initializes the Validator class.

        Args:
          use_reranking: A boolean indicating whether to use reranking.
          validate_documents: A boolean indicating whether to validate documents.
          question_validator_cfg: A configuration dictionary for the QuestionValidator.
          answer_validator_cfg: A configuration dictionary for the AnswerValidator.
          documents_validator_cfg: A configuration dictionary for the DocumentsValidator.
        """
        self.question_validator = (
            QuestionValidator(**question_validator_cfg) if question_validator_cfg is not None else QuestionValidator()
        )
        self.answer_validator = (
            AnswerValidator(**answer_validator_cfg) if answer_validator_cfg is not None else AnswerValidator()
        )
        self.documents_validator = (
            DocumentsValidator(**documents_validator_cfg)
            if documents_validator_cfg is not None
            else DocumentsValidator()
        )
        self.use_reranking = use_reranking
        self.validate_documents = validate_documents

    def check_question_relevance(self, question: str) -> tuple[bool, str]:
        """
        Checks the relevance of a question.

        Args:
          question: The question to be checked.

        Returns:
          A tuple containing a boolean indicating the relevance and a string describing the result.
        """
        return self.question_validator.check_question_relevance(question)

    def check_answer_relevance(self, answer: str) -> bool:
        """
        Checks the relevance of an answer.

        Args:
          answer: The answer to be checked.

        Returns:
          A boolean indicating the relevance of the answer.
        """
        return self.answer_validator.check_answer_relevance(answer)

    def check_documents_relevance(self, answer: str, matched_documents: pd.DataFrame) -> pd.DataFrame:
        """
        Checks the relevance of documents.

        Args:
          answer: The answer to be checked.
          matched_documents: The DataFrame containing the matched documents.

        Returns:
          A DataFrame containing the relevance of the documents.
        """
        return self.documents_validator.check_documents_relevance(answer, matched_documents)

    def rerank_docs(
        self, answer: str, matched_documents: pd.DataFrame, embedding_fn=get_openai_embedding
    ) -> pd.DataFrame:
        """
        Reranks the matched documents based on answer similarity.

        Args:
          answer: The answer for reranking.
          matched_documents: The DataFrame containing the matched documents.
          embedding_fn: The function used to calculate document embeddings.

        Returns:
          A DataFrame containing the reranked documents.
        """
        """Here we re-rank matched documents according to the answer provided by the llm.

        This score could be used to determine wether a document was actually relevant to generation.
        An extra column is added in-place for the similarity score.
        """
        if len(matched_documents) == 0:
            return matched_documents
        logger.info("Reranking documents based on answer similarity...")

        answer_embedding = embedding_fn(answer)

        col = "similarity_to_answer"
        matched_documents[col] = matched_documents.embedding.apply(lambda x: cosine_similarity(x, answer_embedding))

        return matched_documents.sort_values(by=col, ascending=False)
