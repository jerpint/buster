from typing import Iterator
from comet_llm import Span

import openai

from buster.completers import Completer


class ChatGPTCompleter(Completer):
    def complete(self, prompt: str, user_input, completion_kwargs=None) -> str | Iterator:
        # Uses default configuration if not overriden
        if completion_kwargs is None:
            completion_kwargs = self.completion_kwargs

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ]
        stream = completion_kwargs.get("stream")

        inputs = completion_kwargs.copy()
        inputs["messages"] = messages

        with Span(inputs, "llm-call", metadata={"stream": stream}) as span:
            response = openai.ChatCompletion.create(
                messages=messages,
                **completion_kwargs,
            )

            if stream is True:
                # We are entering streaming mode, so here were just wrapping the streamed
                # openai response to be easier to handle later
                def answer_generator():
                    for chunk in response:
                        token: str = chunk["choices"][0]["delta"].get("content", "")
                        yield token

                span.set_outputs(outputs={"output": "stream"})
                return answer_generator()

            else:
                full_response: str = response["choices"][0]["message"]["content"]
                span.set_outputs(outputs={"output": full_response})
                return full_response
