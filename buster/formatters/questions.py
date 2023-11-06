import logging

from buster.completers import ChatGPTCompleter


class QuestionReformulator:
    def __init__(self):
        completion_kwargs = {
            "model": "gpt-3.5-turbo",
            "stream": False,
            "temperature": 0,
        }
        self.completer = ChatGPTCompleter(completion_kwargs=completion_kwargs)
        self.system_prompt = """
        Your role is to reformat a user's input into a question that is useful in the context of a semantic retrieval system.
        Reformulate the question in a way that captures the original essence of the question while also adding more relevant details that can be useful in the context of semantic retrieval."""

    def reformulate(self, user_input: str) -> str:
        """Reformulate a user's question"""
        reformulated_question, error = self.completer.complete(self.system_prompt, user_input=user_input)
        logging.info(f"Reformulated question from {user_input=} to {reformulated_question=}")
        return reformulated_question, error
