import io
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

import pandas as pd
from fastapi.encoders import jsonable_encoder

from buster.completers.user_inputs import UserInputs
from buster.formatters.documents import DocumentsFormatter
from buster.formatters.prompts import PromptFormatter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Completion:
    """
    A class to represent the completion object of a model's output for a user's question.

    Attributes:
        error (bool): A boolean indicating if an error occurred when generating the completion.
        user_inputs (UserInputs): The inputs from the user.
        matched_documents (pd.DataFrame): The documents that were matched to the user's question.
        answer_generator (Iterator): An optional iterator used to generate the model's answer.
        answer_text (str): An optional answer text.
        answer_relevant (bool): An optional boolean indicating if the answer is relevant.
        question_relevant (bool): An optional boolean indicating if the question is relevant.
        completion_kwargs (dict): Optional arguments for the completion.
        validator (Validator): An optional Validator object.

    Methods:
        __repr__: Outputs a string representation of the object.
        _validate_arguments: Validates answer_generator and answer_text arguments.
        answer_relevant: Determines if the answer is relevant or not.
        question_relevant: Retrieves the relevance of the question.
        answer_text: Retrieves the answer text.
        answer_generator: Retrieves the answer generator.
        postprocess: Postprocesses the results after generating the model's answer.
        to_json: Outputs selected attributes of the object in JSON format.
        from_dict: Creates a Completion object from a dictionary.
    """

    def __init__(
        self,
        error: bool,
        user_inputs: UserInputs,
        matched_documents: pd.DataFrame,
        answer_generator: Optional[Iterator] = None,
        answer_text: Optional[str] = None,
        answer_relevant: Optional[bool] = None,
        question_relevant: Optional[bool] = None,
        completion_kwargs: Optional[dict] = None,
        validator=None,
    ):
        self.error = error
        self.user_inputs = user_inputs
        self.matched_documents = matched_documents
        self.validator = validator
        self.completion_kwargs = completion_kwargs
        self._answer_relevant = answer_relevant
        self._question_relevant = question_relevant

        self._validate_arguments(answer_generator, answer_text)

    def __repr__(self):
        class_name = type(self).__name__
        return (
            f"{class_name}("
            f"user_inputs={self.user_inputs!r}, "
            f"error={self.error!r}, "
            f"matched_documents={self.matched_documents!r}, "
            f"answer_text={self._answer_text!r}, "
            f"answer_generator={self.answer_generator!r}, "
            f"answer_relevant={self._answer_relevant!r}, "
            f"question_relevant={self.question_relevant!r}, "
            f"completion_kwargs={self.completion_kwargs!r}, "
            "),"
        )

    def _validate_arguments(self, answer_generator: Optional[Iterator], answer_text: Optional[str]):
        """Sets answer_generator and answer_text properties depending on the provided inputs.

        Checks that one of either answer_generator or answer_text is not None.
        If answer_text is set, a generator can simply be inferred from answer_text.
        If answer_generator is set, answer_text will be set only once the generator gets called. Set to None for now.
        """
        if (answer_generator is None and answer_text is None) or (
            answer_generator is not None and answer_text is not None
        ):
            raise ValueError("Only one of 'answer_generator' and 'answer_text' must be set.")

        # If text is provided, the genrator can be inferred
        if answer_text is not None:
            assert isinstance(answer_text, str)
            answer_generator = (msg for msg in answer_text)

        self._answer_text = answer_text
        self._answer_generator = answer_generator

    @property
    def answer_relevant(self) -> bool:
        """Property determining the relevance of an answer (bool).

        If an error occured, the relevance is False.
        If no documents were retrieved, the relevance is also False.
        Otherwise, the relevance is computed as defined by the validator (e.g. comparing to embeddings)
        """
        if self.error:
            self._answer_relevant = False
        elif len(self.matched_documents) == 0:
            self._answer_relevant = False
        elif self._answer_relevant is not None:
            return self._answer_relevant
        else:
            # Check the answer relevance by looking at the embeddings
            self._answer_relevant = self.validator.check_answer_relevance(self.answer_text)
        return self._answer_relevant

    @property
    def question_relevant(self):
        """Property determining the relevance of the question asked (bool)."""
        return self._question_relevant

    @property
    def answer_text(self):
        if self._answer_text is None:
            # generates the text if it wasn't already generated
            self._answer_text = "".join([i for i in self.answer_generator])
        return self._answer_text

    @answer_text.setter
    def answer_text(self, value: str) -> None:
        self._answer_text = value

    @property
    def answer_generator(self):
        # keeps track of the yielded text
        self._answer_text = ""
        for token in self._answer_generator:
            self._answer_text += token
            yield token

        self.postprocess()

    @answer_generator.setter
    def answer_generator(self, generator: Iterator) -> None:
        self._answer_generator = generator

    def postprocess(self):
        """Function executed after the answer text is generated by the answer_generator"""

        if self.validator is None:
            # TODO: This should only happen if declaring a Completion using .from_dict() method.
            # This behaviour is not ideal and we may want to remove support for .from_dict() in the future.
            logger.info("No validator was set, skipping postprocessing.")
            return

        if self.validator.use_reranking:
            # rerank docs in order of cosine similarity to the question
            self.matched_documents = self.validator.rerank_docs(
                answer=self.answer_text, matched_documents=self.matched_documents
            )

        if self.validator.validate_documents:
            self.matched_documents = self.validator.check_documents_relevance(
                answer=self.answer_text, matched_documents=self.matched_documents
            )

        # access the property so it gets set if not computed alerady
        self.answer_relevant

    def to_json(self, columns_to_ignore: Optional[list[str]] = None) -> Any:
        """Converts selected attributes of the object to a JSON format.

        Args:
            columns_to_ignore (list[str]): A list of column names to ignore in the csulting matched_documents dataframe.

        Returns:
            Any: The object's attributes encoded as JSON.

        Notes:
            - The 'matched_documents' attribute of type pd.DataFrame is encoded separately
            using a custom encoder.
            - The resulting JSON may exclude specified columns based on the 'columns_to_ignore' parameter.
        """

        def encode_df(df: pd.DataFrame) -> dict:
            if columns_to_ignore is not None:
                df = df.drop(columns=columns_to_ignore, errors="ignore")
            return df.to_json(orient="index")

        custom_encoder = {
            # Converts the matched_documents in the user_responses to json
            pd.DataFrame: encode_df,
        }

        to_encode = {
            "user_inputs": self.user_inputs,
            "answer_text": self.answer_text,
            "matched_documents": self.matched_documents,
            "answer_relevant": self.answer_relevant,
            "question_relevant": self.question_relevant,
            "completion_kwargs": self.completion_kwargs,
            "error": self.error,
        }
        return jsonable_encoder(to_encode, custom_encoder=custom_encoder)

    @classmethod
    def from_dict(cls, completion_dict: dict):
        # Map a dict of user inputs to the UserInputs class
        if isinstance(completion_dict["user_inputs"], dict):
            completion_dict["user_inputs"] = UserInputs(**completion_dict["user_inputs"])

        # Map the matched documents back to a dataframe
        if isinstance(completion_dict["matched_documents"], str):
            # avoids deprecation warning
            json_data = io.StringIO(completion_dict["matched_documents"])

            completion_dict["matched_documents"] = pd.read_json(json_data, orient="index")
        elif isinstance(completion_dict["matched_documents"], dict):
            completion_dict["matched_documents"] = pd.DataFrame(completion_dict["matched_documents"]).T
        else:
            raise ValueError(f"Unknown type for matched_documents: {type(completion_dict['matched_documents'])}")

        return cls(**completion_dict)


class Completer(ABC):
    """
    Abstract base class for completers, which generate an answer to a prompt.

    Methods:
        complete: The method that should be implemented by any child class to provide an answer to a prompt.
    """

    @abstractmethod
    def complete(self, prompt: str, user_input) -> (str | Iterator, bool):
        """Returns the completed message (can be a generator), and a boolean to indicate if an error occured or not."""
        ...


class DocumentAnswerer:
    """
    A class that answers questions based on documents.

    It takes care of formatting the prompts and the documents, and generating the answer when relevant.

    Attributes:
        completer (Completer): Object that actually generates an answer to the prompt.
        documents_formatter (DocumentsFormatter): Object that formats the documents for the prompt.
        prompt_formatter (PromptFormatter): Object that prepares the prompt for the completer.
        no_documents_message (str): Message to display when no documents are found to match the query.
        completion_class (Completion): Class to use for the resulting completion.

    Methods:
        prepare_prompt: Prepares the prompt that will be passed to the completer.
        get_completion: Generates a completion to the user's question based on matched documents.
    """

    def __init__(
        self,
        documents_formatter: DocumentsFormatter,
        prompt_formatter: PromptFormatter,
        completer: Completer,
        completion_class: Completion = Completion,
        no_documents_message: str = "No documents were found that match your question.",
    ):
        self.completer = completer
        self.documents_formatter = documents_formatter
        self.prompt_formatter = prompt_formatter
        self.no_documents_message = no_documents_message
        self.completion_class = completion_class

    def prepare_prompt(self, matched_documents) -> str:
        """Prepare the prompt with prompt engineering.

        A user's question is not included here. We use the documents formatter and prompt formatter to
        compose the prompt itself.
        """

        # format the matched documents, (will truncate them if too long)
        formatted_documents, _ = self.documents_formatter.format(matched_documents)
        prompt = self.prompt_formatter.format(formatted_documents)
        return prompt

    def get_completion(
        self,
        user_inputs: UserInputs,
        matched_documents: pd.DataFrame,
        validator,
        question_relevant: bool = True,
    ) -> Completion:
        """Generate a completion to a user's question based on matched documents.

        It is safe to assume the question_relevance to be True if we made it here."""

        logger.info(f"{user_inputs=}")

        if len(matched_documents) == 0:
            warning_msg = "No documents found during retrieval."
            warnings.warn(warning_msg)
            logger.warning(warning_msg)

            # empty dataframe
            matched_documents = pd.DataFrame(columns=matched_documents.columns)

            # because we are requesting a completion, we assume the question is relevant.
            # However, no documents were found, so we pass the no documents found message instead of generating the answer.
            # The completion does not get triggered, so we do not pass completion kwargs here either.
            completion = self.completion_class(
                user_inputs=user_inputs,
                answer_text=self.no_documents_message,
                error=False,
                matched_documents=matched_documents,
                question_relevant=question_relevant,
                validator=validator,
            )
            return completion

        # prepare the prompt with matched documents
        prompt = self.prepare_prompt(matched_documents)
        logger.info(f"{prompt=}")

        logger.info(f"querying model with parameters: {self.completer.completion_kwargs}...")

        try:
            answer_generator, error = self.completer.complete(prompt=prompt, user_input=user_inputs.current_input)

        except Exception as e:
            error = True
            answer_generator = "Something went wrong with the request, try again soon!"
            logger.exception("Unknown error when attempting to generate response. See traceback:")

        completion = self.completion_class(
            answer_generator=answer_generator,
            error=error,
            matched_documents=matched_documents,
            user_inputs=user_inputs,
            question_relevant=question_relevant,
            validator=validator,
            completion_kwargs=self.completer.completion_kwargs,
        )

        return completion
