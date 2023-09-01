from typing import Iterator

from buster.completers import Completer
import os

import together
together.api_key = os.getenv("TOGETHER_API_KEY")
assert together.api_key is not None

together_api_key = os.environ.get("TOGETHER_API_KEY")
assert together_api_key is not None, "check togather API key."

class TogetherAI(Completer):
    def complete(self, prompt: str, user_input, completion_kwargs=None) -> str | Iterator:
        # Uses default configuration if not overriden
        if completion_kwargs is None:
            completion_kwargs = self.completion_kwargs

        # Add the prompt to the completion parameters
        prompt = f"""system prompt: {prompt}\n\n{user_input}\nANSWER:"""
        completion_kwargs["prompt"] = prompt

        # Remove the stream key
        stream = completion_kwargs.pop("stream", False)

        if stream is True:
            response = together.Complete.create_streaming(**completion_kwargs)
            # Here response is already conveniently a string generator
            return response

        else:
            response = together.Complete.create(**completion_kwargs)
            response_text = response['output']['choices'][0]['text']
            return response_text
