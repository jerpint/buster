import logging
import os
from typing import Iterator

import openai

from buster.completers import Completer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Check if an API key exists for promptlayer, if it does, use it
promptlayer_api_key = os.environ.get("PROMPTLAYER_API_KEY")
if promptlayer_api_key:
    try:
        import promptlayer

        logger.info("Enabling prompt layer...")
        promptlayer.api_key = promptlayer_api_key

        # replace openai with the promptlayer wrapper
        openai = promptlayer.openai
    except Exception as e:
        logger.exception("Something went wrong enabling promptlayer.")


class ChatGPTCompleter(Completer):
    def complete(self, prompt: str, user_input: str, completion_kwargs=None) -> (str | Iterator, bool):
        """Returns the completed message (can be a generator), and a boolean to indicate if an error occured or not."""
        # Uses default configuration if not overriden
        if completion_kwargs is None:
            completion_kwargs = self.completion_kwargs

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ]

        try:
            error = False
            response = openai.ChatCompletion.create(
                messages=messages,
                **completion_kwargs,
            )
        except openai.error.InvalidRequestError:
            error = True
            logger.exception("Invalid request to OpenAI API. See traceback:")
            error_message = "Something went wrong with connecting with OpenAI, try again soon!"
            return error_message, error

        except openai.error.RateLimitError:
            error = True
            logger.exception("RateLimit error from OpenAI. See traceback:")
            error_message = "OpenAI servers seem to be overloaded, try again later!"
            return error_message, error

        if completion_kwargs.get("stream") is True:
            # We are entering streaming mode, so here were just wrapping the streamed
            # openai response to be easier to handle later
            def answer_generator():
                for chunk in response:
                    token: str = chunk["choices"][0]["delta"].get("content", "")
                    yield token

            return answer_generator(), error

        else:
            full_response: str = response["choices"][0]["message"]["content"]
            return full_response, error
