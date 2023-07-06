import logging

from openai.embeddings_utils import cosine_similarity

from buster.completers import ChatGPTCompleter
from buster.validators import Validator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QuestionAnswerValidator(Validator):
    def __init__(
        self, completion_kwargs: dict, check_question_prompt: str, unknown_response_templates: list[str], **kwargs
    ):
        super().__init__(**kwargs)

        self.completer = ChatGPTCompleter(completion_kwargs=completion_kwargs)
        self.check_question_prompt = check_question_prompt
        self.unknown_response_templates = unknown_response_templates

    def check_question_relevance(self, question: str) -> tuple[bool, str]:
        """Determines wether a question is relevant or not for our given framework."""

        def get_relevance(outputs: str) -> bool:
            # remove trailing periods, happens sometimes...
            outputs = outputs.strip(".").lower()

            if outputs == "true":
                relevance = True
            elif outputs == "false":
                relevance = False
            else:
                logger.warning(f"the question validation returned an unexpeced value: {outputs}. Assuming Invalid...")
                relevance = False
            return relevance

        response = self.invalid_question_response
        try:
            outputs = self.completer.complete(self.check_question_prompt, user_input=question)
            relevance = get_relevance(outputs)

        except Exception as e:
            logger.exception("Something went wrong during question relevance detection. See traceback:")
            relevance = False
            response = "Unable to process your question at the moment, try again soon"

        logger.info(f"Question {relevance=}")

        return relevance, response

    def check_answer_relevance(self, answer: str) -> bool:
        """Check to see if a generated answer is relevant to the chatbot's knowledge or not.

        We assume we've prompt-engineered our bot to say a response is unrelated to the context if it isn't relevant.
        Here, we compare the embedding of the response to the embedding of the prompt-engineered "I don't know" embedding.

        unk_threshold can be a value between [-1,1]. Usually, 0.85 is a good value.
        """
        logger.info("Checking for answer relevance...")

        if answer == "":
            raise ValueError("Cannot compute embedding of an empty string.")

        # if unknown_prompt is None:
        unknown_responses = self.unknown_response_templates

        unknown_embeddings = [
            self.get_embedding(
                unknown_response,
                engine=self.embedding_model,
            )
            for unknown_response in unknown_responses
        ]

        answer_embedding = self.get_embedding(
            answer,
            engine=self.embedding_model,
        )
        unknown_similarity_scores = [
            cosine_similarity(answer_embedding, unknown_embedding) for unknown_embedding in unknown_embeddings
        ]
        logger.info(f"{unknown_similarity_scores=}")

        # Likely that the answer is meaningful, add the top sources
        answer_relevant: bool = (
            False if any(score > self.unknown_threshold for score in unknown_similarity_scores) else True
        )
        return answer_relevant
