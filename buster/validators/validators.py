import concurrent.futures
import logging
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from buster.completers import ChatGPTCompleter, Completer
from buster.llm_utils import cosine_similarity
from buster.llm_utils.embeddings import get_openai_embedding

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QuestionValidator:
    def __init__(
        self,
        check_question_prompt: Optional[str] = None,
        invalid_question_response: Optional[str] = None,
        completion_kwargs: Optional[dict] = None,
        client_kwargs: Optional[dict] = None,
    ):
        if check_question_prompt is None:
            check_question_prompt = (
                """You are a chatbot answering questions on documentation.
Your job is to determine whether or not a question is valid, and should be answered.
More general questions are not considered valid, even if you might know the response.
A user will submit a question. Respond 'true' if it is valid, respond 'false' if it is invalid.

For example:

Q: What is backpropagation?
true

Q: What is the meaning of life?
false

A user will submit a question. Respond 'true' if it is valid, respond 'false' if it is invalid.""",
            )

        if completion_kwargs is None:
            # default completion kwargs
            completion_kwargs = (
                {
                    "model": "gpt-3.5-turbo",
                    "stream": False,
                    "temperature": 0,
                },
            )

        self.completer = ChatGPTCompleter(completion_kwargs=completion_kwargs, client_kwargs=client_kwargs)
        self.check_question_prompt = check_question_prompt
        self.invalid_question_response = invalid_question_response

    def check_question_relevance(self, question: str) -> tuple[bool, str]:
        """Determines whether a question is relevant for our given framework."""
        try:
            outputs, _ = self.completer.complete(self.check_question_prompt, user_input=question)
            outputs = outputs.strip(".").lower()
            if outputs not in ["true", "false"]:
                logger.warning(f"the question validation returned an unexpeced value: {outputs=}. Assuming Invalid...")
            relevance = outputs.strip(".").lower() == "true"
            response = self.invalid_question_response

        except Exception as e:
            logger.exception("Error during question relevance detection.")
            relevance = False
            response = "Unable to process your question at the moment, try again soon"

        return relevance, response


class AnswerValidator:
    def __init__(
        self,
        unknown_response_templates: Optional[list[str]] = None,
        unknown_threshold: Optional[float] = None,
        embedding_fn: Callable[[str], np.array] = None,
    ):
        if unknown_threshold is None:
            unknown_threshold = 0.85

        if embedding_fn is None:
            embedding_fn = get_openai_embedding

        if unknown_response_templates is None:
            unknown_response_templates = [
                "I'm sorry, but I am an AI language model trained to assist with questions related to AI. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?",
            ]

        self.embedding_fn = embedding_fn
        self.unknown_response_templates = unknown_response_templates
        self.unknown_threshold = unknown_threshold

    def check_answer_relevance(self, answer: str) -> bool:
        """Check if a generated answer is relevant to the chatbot's knowledge."""
        if answer == "":
            raise ValueError("Cannot compute embedding of an empty string.")

        unknown_embeddings = [
            self.embedding_fn(unknown_response) for unknown_response in self.unknown_response_templates
        ]

        answer_embedding = self.embedding_fn(answer)
        unknown_similarity_scores = [
            cosine_similarity(answer_embedding, unknown_embedding) for unknown_embedding in unknown_embeddings
        ]

        # If any score is above the threshold, the answer is considered not relevant
        return not any(score > self.unknown_threshold for score in unknown_similarity_scores)


class DocumentsValidator:
    def __init__(
        self,
        completion_kwargs: Optional[dict] = None,
        client_kwargs: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        user_input_formatter: Optional[str] = None,
        max_calls: int = 30,
    ):
        if system_prompt is None:
            system_prompt = """
            Your goal is to determine if the content of a document can be attributed to a provided answer.
            This means that if information in the document is found in the answer, it is relevant. Otherwise it is not.
            Your goal is to determine if the information contained in a document was used to generate an answer.
            You will be comparing a document to an answer. If the answer can be inferred from the document, return 'true'. Otherwise return 'false'.
            Only respond with 'true' or 'false'."""
        self.system_prompt = system_prompt

        if user_input_formatter is None:
            user_input_formatter = """
            answer: {answer}
            document: {document}
        """
        self.user_input_formatter = user_input_formatter

        if completion_kwargs is None:
            completion_kwargs = {
                "model": "gpt-3.5-turbo",
                "stream": False,
                "temperature": 0,
            }

        self.completer = ChatGPTCompleter(completion_kwargs=completion_kwargs, client_kwargs=client_kwargs)

        self.max_calls = max_calls

    def check_document_relevance(self, answer: str, document: str) -> bool:
        user_input = self.user_input_formatter.format(answer=answer, document=document)
        output, _ = self.completer.complete(prompt=self.system_prompt, user_input=user_input)

        # remove trailing periods, happens sometimes...
        output = output.strip(".").lower()

        if output not in ["true", "false"]:
            # Default assume it's relevant if the detector didn't give one of [true, false]
            logger.warning(f"the validation returned an unexpected value: {output}. Assuming valid...")
            return True
        return output == "true"

    def check_documents_relevance(self, answer: str, matched_documents: pd.DataFrame) -> list[bool]:
        """Determines wether a question is relevant or not for our given framework."""

        logger.info(f"Checking document relevance of {len(matched_documents)} documents")

        if len(matched_documents) > self.max_calls:
            raise ValueError("Max calls exceeded, increase max_calls to allow this.")

        # Here we parallelize the calls. We introduce a wrapper as a workaround.
        def _check_documents(args):
            "Thin wrapper so we can pass args as a Tuple and use ThreadPoolExecutor."
            answer, document = args
            return self.check_document_relevance(answer=answer, document=document)

        args_list = [(answer, doc) for doc in matched_documents.content.to_list()]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            relevance = list(executor.map(_check_documents, args_list))

        logger.info(f"{relevance=}")
        # add it back to the dataframe
        matched_documents["relevance"] = relevance
        return matched_documents
