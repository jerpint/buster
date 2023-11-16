import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from buster.completers import Completion, DocumentAnswerer, UserInputs
from buster.llm_utils import QuestionReformulator, get_openai_embedding
from buster.retriever import Retriever
from buster.validators import Validator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class BusterConfig:
    """Configuration object for a chatbot."""

    validator_cfg: dict = field(
        default_factory=lambda: {
            "use_reranking": True,
            "validate_documents": False,
        }
    )
    tokenizer_cfg: dict = field(
        default_factory=lambda: {
            "model_name": "gpt-3.5-turbo",
        }
    )
    retriever_cfg: dict = field(
        default_factory=lambda: {
            "max_tokens": 3000,
            "top_k": 3,
            "thresh": 0.7,
            "embedding_fn": get_openai_embedding,
        }
    )
    prompt_formatter_cfg: dict = field(
        default_factory=lambda: {
            "max_tokens": 3500,
            "text_before_docs": "You are a chatbot answering questions.\n",
            "text_after_docs": "Answer the following question:\n",
            "formatter": "{text_before_docs}\n{documents}\n{text_after_docs}",
        }
    )
    documents_formatter_cfg: dict = (
        field(
            default_factory=lambda: {
                "max_tokens": 3500,
                "formatter": "{content}",
            }
        ),
    )
    documents_answerer_cfg: dict = field(
        default_factory=lambda: {
            "no_documents_message": "No documents are available for this question.",
        }
    )
    question_reformulator_cfg: dict = field(
        default_factory=lambda: {
            "completion_kwargs": {
                "model": "gpt-3.5-turbo",
                "stream": False,
                "temperature": 0,
            },
            "system_prompt": """
            Your role is to reformat a user's input into a question that is useful in the context of a semantic retrieval system.
            Reformulate the question in a way that captures the original essence of the question while also adding more relevant details that can be useful in the context of semantic retrieval.""",
        }
    )
    completion_cfg: dict = field(
        default_factory=lambda: {
            "completion_kwargs": {
                "model": "gpt-3.5-turbo",
                "temperature": 0,
                "stream": True,
            },
        }
    )


class Buster:
    def __init__(
        self,
        retriever: Retriever,
        document_answerer: DocumentAnswerer,
        validator: Validator,
        question_reformulator: Optional[QuestionReformulator] = None,
    ):
        self.document_answerer = document_answerer
        self.retriever = retriever
        self.validator = validator
        self.question_reformulator = question_reformulator

    def process_input(
        self,
        user_input: str,
        sources: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        reformulate_question: Optional[bool] = False,
    ) -> Completion:
        """
        Main function to process the input question and generate a formatted output.
        """

        logger.info(f"User Input:\n{user_input}")

        # We make sure there is always a newline at the end of the question to avoid completing the question.
        if not user_input.endswith("\n"):
            user_input += "\n"

        user_inputs = UserInputs(original_input=user_input)

        # The returned message is either a generic invalid question message or an error handling message
        question_relevant, irrelevant_question_message = self.validator.check_question_relevance(user_input)

        if question_relevant:
            # question is relevant, get completor to generate completion

            # reformulate the question if a reformulator is defined
            if self.question_reformulator is not None and reformulate_question:
                reformulated_input, reformulation_error = self.question_reformulator.reformulate(
                    user_inputs.original_input
                )
                user_inputs.reformulated_input = reformulated_input

                if reformulation_error:
                    completion = Completion(
                        error=True,
                        user_inputs=user_inputs,
                        matched_documents=pd.DataFrame(),
                        answer_text="Something went wrong reformulating the question. Try again soon.",
                        answer_relevant=False,
                        question_relevant=False,
                        validator=self.validator,
                    )
                    return completion

            # Retrieve and answer
            matched_documents = self.retriever.retrieve(user_inputs, sources=sources, top_k=top_k)
            completion: Completion = self.document_answerer.get_completion(
                user_inputs=user_inputs,
                matched_documents=matched_documents,
                validator=self.validator,
                question_relevant=question_relevant,
            )
            return completion

        else:
            # question was determined irrelevant, so we instead return a generic response set by the user.
            completion = Completion(
                error=False,
                user_inputs=user_inputs,
                matched_documents=pd.DataFrame(),
                answer_text=irrelevant_question_message,
                answer_relevant=False,
                question_relevant=False,
                validator=self.validator,
            )
            return completion
