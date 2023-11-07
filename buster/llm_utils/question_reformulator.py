import logging
from typing import Optional

from buster.completers import ChatGPTCompleter


class QuestionReformulator:
    def __init__(self, system_prompt: Optional[str] = None, completion_kwargs: Optional[dict] = None):
        self.completer = ChatGPTCompleter(completion_kwargs=completion_kwargs)

        if completion_kwargs is None:
            # Default kwargs
            completion_kwargs = {
                "model": "gpt-3.5-turbo",
                "stream": False,
                "temperature": 0,
            }
        self.completion_kwargs = completion_kwargs

        if system_prompt is None:
            # Default prompt
            system_prompt = """
            Your role is to reformat a user's input into a question that is useful in the context of a semantic retrieval system.
            Reformulate the question in a way that captures the original essence of the question while also adding more relevant details that can be useful in the context of semantic retrieval."""
        self.system_prompt = system_prompt

    def reformulate(self, user_input: str) -> str:
        """Reformulate a user's question"""
        reformulated_question, error = self.completer.complete(
            self.system_prompt, user_input=user_input, completion_kwargs=self.completion_kwargs
        )
        logging.info(f"Reformulated question from {user_input=} to {reformulated_question=}")
        return reformulated_question, error
